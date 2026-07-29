# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Benchmark GDS vs uGDS async write at realistic chunk size (32MB).

Measures write throughput through the full GDSContext path and raw backend,
complementing bench_chunk_read.py.
"""

# Standard
import importlib.util
import os
import time

# Third Party
import torch

CHUNK = 32 * 1024 * 1024  # 32MB
DEPTHS = [int(d) for d in os.environ.get("LMCACHE_BENCH_DEPTHS", "1,4,16").split(",")]
WARMUP = int(os.environ.get("LMCACHE_BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("LMCACHE_BENCH_ITERS", "20"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_ugds_device():
    for i in range(4):
        p = f"/dev/ugds_drv{i}"
        if os.path.exists(p):
            return p
    raise RuntimeError("no ugds_drv device")


def chunk_pattern(chunk_index):
    return (chunk_index % 255) + 1


def bench_write(handle, buffers, depth, raw_stream):
    latencies = []
    size = CHUNK
    for i in range(WARMUP + ITERS):
        for idx, buf in enumerate(buffers[:depth]):
            buf.fill_(chunk_pattern(idx))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        subs = []
        for d in range(depth):
            sub = handle.write_async(
                buffers[d].data_ptr(),
                size,
                file_offset=d * size,
                buf_offset=0,
                raw_stream=raw_stream,
            )
            subs.append(sub)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        for idx, sub in enumerate(subs):
            if sub.bytes_done != size:
                raise RuntimeError(
                    f"short write chunk {idx}: expected {size}, got {sub.bytes_done}"
                )
        del subs
        if i >= WARMUP:
            latencies.append((t1 - t0) * 1e6)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = size * depth
    bw = total_bytes / (median / 1e6) / (1024 * 1024)
    per_io = median / depth
    return median, per_io, bw


def verify_readback(handle, buffers, depth, raw_stream):
    size = CHUNK
    for buf in buffers[:depth]:
        buf.zero_()
    torch.cuda.synchronize()
    subs = []
    for d in range(depth):
        sub = handle.read_async(
            buffers[d].data_ptr(),
            size,
            file_offset=d * size,
            buf_offset=0,
            raw_stream=raw_stream,
        )
        subs.append(sub)
    torch.cuda.synchronize()
    del subs
    for idx, buf in enumerate(buffers[:depth]):
        pattern = chunk_pattern(idx)
        mismatch = torch.count_nonzero(buf != pattern).item()
        if mismatch:
            raise RuntimeError(
                f"readback mismatch chunk {idx}: pattern=0x{pattern:02x}, "
                f"mismatched={mismatch}/{size}"
            )


def bench_context_write(context, buffers, memory_objects, depth):
    # First Party
    from lmcache.v1.gpu_connector.gds_context import SlabDirection

    latencies = []
    for iteration in range(WARMUP + ITERS):
        for idx, buf in enumerate(buffers[:depth]):
            buf.fill_(chunk_pattern(idx))
        torch.cuda.synchronize()

        start = time.perf_counter()
        for chunk_index in range(depth):
            context.transfer_async(
                memory_objects[chunk_index],
                buffers[chunk_index],
                SlabDirection.WRITE,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        if iteration >= WARMUP:
            latencies.append((end - start) * 1e6)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = CHUNK * depth
    bw = total_bytes / (median / 1e6) / (1024 * 1024)
    return median, median / depth, bw


def run_ugds():
    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "lmcache",
        "v1",
        "gpu_connector",
        "_ugds_async.py",
    )
    ua = _load_module("_ugds_async", os.path.normpath(base))

    device_path = find_ugds_device()
    print(f"uGDS device: {device_path}")

    fd = os.open(device_path, os.O_RDWR)
    ugds_handle = ua.register_handle(fd)
    handle = ua.AsyncHandle.from_fd(fd, ugds_handle, device_path, writable=True)
    buffers = [
        torch.empty(CHUNK, dtype=torch.uint8, device="cuda") for _ in range(max(DEPTHS))
    ]
    for buf in buffers:
        ua.register_buffer(buf)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ua.register_stream(raw_stream)

    print(
        f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
        f"{'BW(MB/s)':>10s}  {'Data':>8s}"
    )
    print("-" * 58)
    for depth in DEPTHS:
        med, per_io, bw = bench_write(handle, buffers, depth, raw_stream)
        verify_readback(handle, buffers, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}  {'PASS':>8s}")

    ua.deregister_stream(raw_stream)
    for buf in buffers:
        ua.deregister_buffer(buf)
    handle.close()


def run_gds(gds_file):
    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "lmcache",
        "v1",
        "gpu_connector",
        "_cufile_async.py",
    )
    ca = _load_module("_cufile_async", os.path.normpath(base))

    max_total = CHUNK * max(DEPTHS)
    fd_create = os.open(gds_file, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.posix_fallocate(fd_create, 0, max_total + 4096)
    os.close(fd_create)

    fd = os.open(gds_file, os.O_RDWR | os.O_DIRECT)
    cufile_handle = ca.register_handle(fd)
    handle = ca.AsyncHandle.from_fd(fd, cufile_handle, gds_file, writable=True)

    buffers = [
        torch.empty(CHUNK, dtype=torch.uint8, device="cuda") for _ in range(max(DEPTHS))
    ]
    for buf in buffers:
        ca.register_buffer(buf)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ca.register_stream(raw_stream)

    print(
        f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
        f"{'BW(MB/s)':>10s}  {'Data':>8s}"
    )
    print("-" * 58)
    for depth in DEPTHS:
        med, per_io, bw = bench_write(handle, buffers, depth, raw_stream)
        verify_readback(handle, buffers, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}  {'PASS':>8s}")

    ca.deregister_stream(raw_stream)
    for buf in buffers:
        ca.deregister_buffer(buf)
    handle.close()


def run_context(backend):
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.config import GdsL1Config
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
    from lmcache.v1.gpu_connector.gds_context import GDSContext

    location = find_ugds_device() if backend == "ugds" else "/mnt/ugds_test"
    config = GdsL1Config(
        file_location=location,
        size_in_bytes=CHUNK * max(DEPTHS),
        backend=backend,
    )
    context = GDSContext()
    context.initialize(config)
    buffers = [
        torch.empty(CHUNK, dtype=torch.uint8, device="cuda") for _ in range(max(DEPTHS))
    ]
    for buf in buffers:
        context.register_gpu_buffer(buf)

    manager = GDSL1MemoryManager(config)
    error, memory_objects = manager.allocate(
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK])], dtypes=[torch.uint8]),
        max(DEPTHS),
    )
    if error != L1Error.SUCCESS:
        raise RuntimeError(f"allocate failed: {error}")

    try:
        print(
            f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
            f"{'BW(MB/s)':>10s}"
        )
        print("-" * 48)
        for depth in DEPTHS:
            med, per_io, bw = bench_context_write(
                context, buffers, memory_objects, depth
            )
            print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}")
    finally:
        for buf in buffers:
            context.deregister_gpu_buffer(buf)
        context.close()


if __name__ == "__main__":
    # Standard
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "ugds"
    print(f"=== 32MB Chunk Write Benchmark: {backend.upper()} ===")
    if backend == "gds":
        run_gds("/mnt/ugds_test/bench_chunk_write.bin")
    elif backend in ("gds-context", "ugds-context"):
        run_context("ugds" if backend == "ugds-context" else "cufile")
    else:
        run_ugds()
