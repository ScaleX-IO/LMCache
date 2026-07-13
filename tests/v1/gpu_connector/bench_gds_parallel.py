#!/usr/bin/env python3
"""Test whether GDS parallel_io kicks in at >= 8MB (min_io_threshold_size_kb).

Compare async read latency/BW at sizes below and above the 8MB threshold,
at depth=1 (single IO) to isolate the parallel_io split effect.
"""

import importlib.util
import os
import time
import torch

SIZES = [
    1 * 1024 * 1024,    # 1M  - below threshold
    2 * 1024 * 1024,    # 2M
    4 * 1024 * 1024,    # 4M
    8 * 1024 * 1024,    # 8M  - at threshold
    16 * 1024 * 1024,   # 16M - above threshold
    32 * 1024 * 1024,   # 32M
]
WARMUP = 3
ITERS = 20


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "lmcache", "v1", "gpu_connector", "_cufile_async.py",
    )
    ca = _load_module("_cufile_async", os.path.normpath(base))

    gds_file = "/mnt/ugds_test/bench_parallel.bin"
    max_size = max(SIZES)

    fd_create = os.open(gds_file, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.posix_fallocate(fd_create, 0, max_size + 4096)
    os.close(fd_create)

    flags = os.O_RDWR | os.O_DIRECT
    fd = os.open(gds_file, flags)
    cufile_handle = ca.register_handle(fd)
    handle = ca.AsyncHandle.from_fd(fd, cufile_handle, gds_file, writable=True)

    buf = torch.empty(max_size, dtype=torch.uint8, device="cuda")
    ca.register_buffer(buf)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ca.register_stream(raw_stream)

    # Pre-fill file with data
    buf.fill_(0x42)
    torch.cuda.synchronize()
    handle.write_async(buf.data_ptr(), max_size, file_offset=0, buf_offset=0,
                       raw_stream=raw_stream)
    torch.cuda.synchronize()

    print(f"GDS parallel_io threshold: 8MB (min_io_threshold_size_kb=8192)")
    print(f"max_io_threads=4, max_request_parallelism=4")
    print(f"Iters: {ITERS} (warmup: {WARMUP})")
    print()
    print(f"{'Size':>6s}  {'Median(us)':>12s}  {'p99(us)':>12s}  {'BW(MB/s)':>10s}  {'Note':s}")
    print("-" * 65)

    for sz in SIZES:
        latencies = []
        for i in range(WARMUP + ITERS):
            buf[:sz].zero_()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            handle.read_async(buf.data_ptr(), sz, file_offset=0, buf_offset=0,
                              raw_stream=raw_stream)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= WARMUP:
                latencies.append((t1 - t0) * 1e6)

        latencies.sort()
        median = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]
        bw = sz / (median / 1e6) / (1024 * 1024)
        note = "<-- threshold" if sz == 8 * 1024 * 1024 else ""
        sz_str = f"{sz // (1024*1024)}M"
        print(f"{sz_str:>6s}  {median:12.1f}  {p99:12.1f}  {bw:10.1f}  {note}")

    ca.deregister_stream(raw_stream)
    ca.deregister_buffer(buf)
    handle.close()


if __name__ == "__main__":
    main()
