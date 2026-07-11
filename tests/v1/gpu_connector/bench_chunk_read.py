#!/usr/bin/env python3
"""Benchmark GDS vs uGDS async read at realistic chunk size (32MB).

Pre-fill uses POSIX writes (avoids cuFileWriteAsync state pollution).
Tests single-IO and pipeline modes to match LMCache gds_context usage.
"""

import importlib
import os
import time
import torch

CHUNK = 32 * 1024 * 1024  # 32MB, Llama3-8B fp16 chunk
DEPTHS = [1, 4, 16]
WARMUP = 3
ITERS = 20


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


def posix_prefill(path, num_chunks):
    """Pre-fill file with POSIX writes + fsync."""
    total = CHUNK * num_chunks
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    data = b'\x42' * CHUNK
    for d in range(num_chunks):
        os.pwrite(fd, data, d * CHUNK)
    os.fsync(fd)
    os.close(fd)
    print(f"POSIX pre-fill: {num_chunks} x 32MB = {total // (1024*1024)}MB")


def bench(handle, buf, depth, raw_stream):
    latencies = []
    size = CHUNK
    for i in range(WARMUP + ITERS):
        buf.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        subs = []
        for d in range(depth):
            sub = handle.read_async(
                buf.data_ptr(), size, file_offset=d * size, buf_offset=0,
                raw_stream=raw_stream,
            )
            subs.append(sub)
        torch.cuda.synchronize()
        del subs
        t1 = time.perf_counter()
        if i >= WARMUP:
            if buf[0].item() == 0:
                print(f"  WARNING: iter {i} depth={depth} buf still zero!")
            latencies.append((t1 - t0) * 1e6)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    total_bytes = size * depth
    bw = total_bytes / (median / 1e6) / (1024 * 1024)
    per_io = median / depth
    return median, per_io, bw


def run_gds(gds_file):
    base = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "lmcache", "v1", "gpu_connector", "_cufile_async.py",
    )
    ca = _load_module("_cufile_async", os.path.normpath(base))

    posix_prefill(gds_file, max(DEPTHS))

    fd = os.open(gds_file, os.O_RDWR | os.O_DIRECT)
    cufile_handle = ca.register_handle(fd)
    handle = ca.AsyncHandle.from_fd(fd, cufile_handle, gds_file, writable=True)

    buf = torch.empty(CHUNK, dtype=torch.uint8, device="cuda")
    ca.register_buffer(buf)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ca.register_stream(raw_stream)

    print(f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  {'BW(MB/s)':>10s}")
    print("-" * 48)
    for depth in DEPTHS:
        med, per_io, bw = bench(handle, buf, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}")

    ca.deregister_stream(raw_stream)
    ca.deregister_buffer(buf)
    handle.close()


def run_ugds():
    base = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "lmcache", "v1", "gpu_connector", "_ugds_async.py",
    )
    ua = _load_module("_ugds_async", os.path.normpath(base))

    device_path = find_ugds_device()
    print(f"uGDS device: {device_path}")

    handle = ua.register_handle(device_path)
    buf = torch.empty(CHUNK, dtype=torch.uint8, device="cuda")
    ua.register_buffer(buf)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ua.register_stream(raw_stream)

    # Pre-fill via uGDS async write (ugds_drv doesn't support POSIX IO)
    buf.fill_(0x42)
    torch.cuda.synchronize()
    subs = []
    for d in range(max(DEPTHS)):
        sub = handle.write_async(buf.data_ptr(), CHUNK, file_offset=d * CHUNK,
                                 buf_offset=0, raw_stream=raw_stream)
        subs.append(sub)
    torch.cuda.synchronize()
    del subs
    print(f"uGDS pre-fill: {max(DEPTHS)} x 32MB = {CHUNK * max(DEPTHS) // (1024*1024)}MB")

    print(f"\n{'Depth':>6s}  {'Total(us)':>12s}  {'Per-IO(us)':>12s}  {'BW(MB/s)':>10s}")
    print("-" * 48)
    for depth in DEPTHS:
        med, per_io, bw = bench(handle, buf, depth, raw_stream)
        print(f"{depth:>6d}  {med:12.1f}  {per_io:12.1f}  {bw:10.1f}")

    ua.deregister_stream(raw_stream)
    ua.deregister_buffer(buf)
    handle.close()


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "gds"
    print(f"=== 32MB Chunk Read Benchmark: {backend.upper()} ===")
    if backend == "gds":
        run_gds("/mnt/ugds_test/bench_chunk.bin")
    else:
        run_ugds()
