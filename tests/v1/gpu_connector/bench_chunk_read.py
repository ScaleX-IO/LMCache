#!/usr/bin/env python3
"""Benchmark GDS vs uGDS async read at realistic chunk size (32MB).

Pre-fill uses POSIX writes (avoids cuFileWriteAsync state pollution).
Tests single-IO and pipeline modes to match LMCache gds_context usage.
"""

import importlib.util
import os
import time
import torch

CHUNK = 32 * 1024 * 1024  # 32MB, Llama3-8B fp16 chunk
DEPTHS = [
    int(depth) for depth in os.environ.get("LMCACHE_BENCH_DEPTHS", "1,4,16").split(",")
]
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


def posix_prefill(path, num_chunks):
    """Pre-fill file with POSIX writes + fsync."""
    total = CHUNK * num_chunks
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    try:
        for chunk_index in range(num_chunks):
            pattern = chunk_pattern(chunk_index)
            data = bytes([pattern]) * CHUNK
            written = os.pwrite(fd, data, chunk_index * CHUNK)
            if written != CHUNK:
                raise RuntimeError(
                    f"short POSIX write for chunk {chunk_index}: "
                    f"expected {CHUNK}, got {written}"
                )
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"POSIX pre-fill: {num_chunks} x 32MB = {total // (1024 * 1024)}MB")


def validate_buffers(buffers, depth):
    for chunk_index, buffer in enumerate(buffers[:depth]):
        pattern = chunk_pattern(chunk_index)
        mismatch_count = torch.count_nonzero(buffer != pattern).item()
        if mismatch_count:
            raise RuntimeError(
                f"data mismatch in chunk {chunk_index}: pattern=0x{pattern:02x}, "
                f"mismatched_bytes={mismatch_count}/{CHUNK}"
            )


def bench(handle, buffers, depth, raw_stream):
    latencies = []
    size = CHUNK
    for i in range(WARMUP + ITERS):
        for buffer in buffers[:depth]:
            buffer.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        for chunk_index, sub in enumerate(subs):
            if sub.bytes_done != size:
                raise RuntimeError(
                    f"short async read for chunk {chunk_index}: "
                    f"expected {size}, got {sub.bytes_done}"
                )
        del subs
        if i >= WARMUP:
            latencies.append((t1 - t0) * 1e6)

    validate_buffers(buffers, depth)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = size * depth
    bw = total_bytes / (median / 1e6) / (1024 * 1024)
    per_io = median / depth
    return median, per_io, bw


def bench_sync(handle, buffers, depth):
    from cufile.bindings import libcufile

    latencies = []
    for iteration in range(WARMUP + ITERS):
        for buffer in buffers[:depth]:
            buffer.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for chunk_index, buffer in enumerate(buffers[:depth]):
            bytes_done = libcufile.cuFileRead(
                handle._handle,
                buffer.data_ptr(),
                CHUNK,
                chunk_index * CHUNK,
                0,
            )
            if bytes_done != CHUNK:
                raise RuntimeError(
                    f"short synchronous read for chunk {chunk_index}: "
                    f"expected {CHUNK}, got {bytes_done}"
                )
        end = time.perf_counter()
        if iteration >= WARMUP:
            latencies.append((end - start) * 1e6)

    validate_buffers(buffers, depth)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = CHUNK * depth
    bandwidth = total_bytes / (median / 1e6) / (1024 * 1024)
    return median, median / depth, bandwidth


def bench_context(context, buffers, memory_objects, depth):
    from lmcache.v1.gpu_connector.gds_context import SlabDirection

    latencies = []
    for iteration in range(WARMUP + ITERS):
        for buffer in buffers[:depth]:
            buffer.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for chunk_index in range(depth):
            context.transfer_async(
                memory_objects[chunk_index],
                buffers[chunk_index],
                SlabDirection.READ,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        if iteration >= WARMUP:
            latencies.append((end - start) * 1e6)

    validate_buffers(buffers, depth)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = CHUNK * depth
    bandwidth = total_bytes / (median / 1e6) / (1024 * 1024)
    return median, median / depth, bandwidth


def run_gds(gds_file, synchronous=False):
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
    stream_flags = os.environ.get("LMCACHE_BENCH_CUFILE_STREAM_FLAGS")
    if stream_flags is not None:
        ca._STREAM_REGISTER_FLAGS = int(stream_flags, 0)
        print(f"cuFile stream registration flags: {ca._STREAM_REGISTER_FLAGS:#x}")

    posix_prefill(gds_file, max(DEPTHS))

    fd = os.open(gds_file, os.O_RDWR | os.O_DIRECT)
    cufile_handle = ca.register_handle(fd)
    handle = ca.AsyncHandle.from_fd(fd, cufile_handle, gds_file, writable=True)

    buffers = [
        torch.empty(CHUNK, dtype=torch.uint8, device="cuda") for _ in range(max(DEPTHS))
    ]
    for buffer in buffers:
        ca.register_buffer(buffer)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ca.register_stream(raw_stream)

    print(
        f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
        f"{'BW(MB/s)':>10s}  {'Data':>8s}"
    )
    print("-" * 58)
    for depth in DEPTHS:
        if synchronous:
            med, per_io, bw = bench_sync(handle, buffers, depth)
        else:
            med, per_io, bw = bench(handle, buffers, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}  {'PASS':>8s}")

    ca.deregister_stream(raw_stream)
    for buffer in buffers:
        ca.deregister_buffer(buffer)
    handle.close()


def run_gds_context(backend):
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.config import GdsL1Config
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
    from lmcache.v1.gpu_connector.gds_context import GDSContext, SlabDirection

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
    for buffer in buffers:
        context.register_gpu_buffer(buffer)

    manager = GDSL1MemoryManager(config)
    error, memory_objects = manager.allocate(
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK])], dtypes=[torch.uint8]),
        max(DEPTHS),
    )
    if error != L1Error.SUCCESS:
        raise RuntimeError(f"failed to allocate uGDS slab chunks: {error}")

    try:
        for chunk_index, buffer in enumerate(buffers):
            buffer.fill_(chunk_pattern(chunk_index))
            context.transfer_async(
                memory_objects[chunk_index],
                buffer,
                SlabDirection.WRITE,
            )
        torch.cuda.synchronize()
        total_mib = CHUNK * max(DEPTHS) // (1024 * 1024)
        print(f"LMCache {backend} pre-fill: {max(DEPTHS)} x 32MB = {total_mib}MB")

        print(
            f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
            f"{'BW(MB/s)':>10s}  {'Data':>8s}"
        )
        print("-" * 58)
        for depth in DEPTHS:
            median, per_io, bandwidth = bench_context(
                context, buffers, memory_objects, depth
            )
            print(
                f"{depth:>6d}  {median:12.1f}  {per_io:12.1f}  "
                f"{bandwidth:10.1f}  {'PASS':>8s}"
            )
    finally:
        cleanup_start = time.perf_counter()
        for buffer in buffers:
            context.deregister_gpu_buffer(buffer)
        context.close()
        cleanup_ms = (time.perf_counter() - cleanup_start) * 1e3
        print(f"GDSContext cleanup: {cleanup_ms:.1f}ms")


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
    for buffer in buffers:
        ua.register_buffer(buffer)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ua.register_stream(raw_stream)

    # Pre-fill via uGDS async write (ugds_drv doesn't support POSIX IO)
    for chunk_index, buffer in enumerate(buffers):
        buffer.fill_(chunk_pattern(chunk_index))
    torch.cuda.synchronize()
    subs = []
    for chunk_index, buffer in enumerate(buffers):
        sub = handle.write_async(
            buffer.data_ptr(),
            CHUNK,
            file_offset=chunk_index * CHUNK,
            buf_offset=0,
            raw_stream=raw_stream,
        )
        subs.append(sub)
    torch.cuda.synchronize()
    for chunk_index, sub in enumerate(subs):
        if sub.bytes_done != CHUNK:
            raise RuntimeError(
                f"short async write for chunk {chunk_index}: "
                f"expected {CHUNK}, got {sub.bytes_done}"
            )
    del subs
    total_mib = CHUNK * max(DEPTHS) // (1024 * 1024)
    print(f"uGDS pre-fill: {max(DEPTHS)} x 32MB = {total_mib}MB")

    print(
        f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  "
        f"{'BW(MB/s)':>10s}  {'Data':>8s}"
    )
    print("-" * 58)
    for depth in DEPTHS:
        med, per_io, bw = bench(handle, buffers, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}  {'PASS':>8s}")

    ua.deregister_stream(raw_stream)
    for buffer in buffers:
        ua.deregister_buffer(buffer)
    handle.close()


if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "gds"
    print(f"=== 32MB Chunk Read Benchmark: {backend.upper()} ===")
    if backend in ("gds", "gds-sync"):
        run_gds(
            "/mnt/ugds_test/bench_chunk.bin",
            synchronous=backend == "gds-sync",
        )
    elif backend in ("gds-context", "ugds-context"):
        run_gds_context("ugds" if backend == "ugds-context" else "cufile")
    else:
        run_ugds()
