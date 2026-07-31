# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Benchmark _ugds_async vs _cufile_async through the LMCache GDS interface.

Pipeline mode: submit DEPTH IOs at different offsets, sync once, measure total
throughput — mirrors how gds_context.py fires many transfer_async() calls on a
CUDA stream before any synchronization.

Usage:
    # Phase 1: test uGDS (990 PRO bound to ugds_drv)
    LD_LIBRARY_PATH=/path/to/uGDS/build \
        python bench_ugds_vs_gds.py --backend ugds

    # Then switch driver:
    #   cd /path/to/uGDS && scripts/env_switch.sh gds 0000:b8:00.0
    #   sudo mount -o data=ordered /dev/nvme0n1 /mnt/ugds_test

    # Phase 2: test GDS
    python bench_ugds_vs_gds.py \
        --backend gds --gds-file /mnt/ugds_test/bench_slab.bin

Results are printed as a table and saved to bench_results_{backend}.json.
"""

# Standard
import argparse
import importlib.util
import json
import os
import time

# Third Party
import torch

SIZES = [4096, 64 * 1024, 128 * 1024, 512 * 1024, 1 * 1024 * 1024]
DEPTHS = [1, 4, 16, 64]
WARMUP = 3
ITERS = 30


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


def bench_pipeline(handle, buf, size, raw_stream, depth, mode="read"):
    """Submit staggered IOs, sync once, and return latency and bandwidth."""
    latencies = []

    for i in range(WARMUP + ITERS):
        if mode == "write":
            buf[:size].fill_(0xAB)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for d in range(depth):
            offset = d * size
            if mode == "read":
                handle.read_async(
                    buf.data_ptr(),
                    size,
                    file_offset=offset,
                    buf_offset=0,
                    raw_stream=raw_stream,
                )
            else:
                handle.write_async(
                    buf.data_ptr(),
                    size,
                    file_offset=offset,
                    buf_offset=0,
                    raw_stream=raw_stream,
                )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= WARMUP:
            latencies.append((t1 - t0) * 1e6)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    total_bytes = size * depth
    bw = total_bytes / (median / 1e6) / (1024 * 1024)
    per_io = median / depth
    return median, p99, per_io, bw


def run_backend(backend_name, mod, handle, buf_size, raw_stream):
    buf = torch.empty(buf_size, dtype=torch.uint8, device="cuda")
    mod.register_buffer(buf)

    results = {}
    for mode in ("read", "write"):
        for sz in SIZES:
            for depth in DEPTHS:
                if sz * depth > buf_size:
                    continue
                med, p99, per_io, bw = bench_pipeline(
                    handle,
                    buf,
                    sz,
                    raw_stream,
                    depth,
                    mode,
                )
                label = f"{mode}_{sz}_d{depth}"
                results[label] = {
                    "size": sz,
                    "mode": mode,
                    "depth": depth,
                    "total_us": med,
                    "p99_us": p99,
                    "per_io_us": per_io,
                    "bw_MBps": bw,
                }
                sz_str = (
                    f"{sz // 1024}K" if sz < 1024 * 1024 else f"{sz // (1024 * 1024)}M"
                )
                print(
                    f"  {backend_name:4s} {mode:5s} {sz_str:>5s} x{depth:<3d}: "
                    f"total={med:10.1f}us  per_io={per_io:8.1f}us  BW={bw:8.1f} MB/s"
                )

    mod.deregister_buffer(buf)
    return results


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

    max_total = max(SIZES) * max(DEPTHS)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ua.register_stream(raw_stream)

    results = run_backend("uGDS", ua, handle, max_total, raw_stream)

    ua.deregister_stream(raw_stream)
    handle.close()
    return results


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

    print(f"GDS file: {gds_file}")
    max_total = max(SIZES) * max(DEPTHS)

    fd_create = os.open(gds_file, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.posix_fallocate(fd_create, 0, max_total + 4096)
    os.close(fd_create)

    flags = os.O_RDWR | os.O_DIRECT
    fd = os.open(gds_file, flags)
    cufile_handle = ca.register_handle(fd)
    handle = ca.AsyncHandle.from_fd(fd, cufile_handle, gds_file, writable=True)

    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    ca.register_stream(raw_stream)

    results = run_backend("GDS", ca, handle, max_total, raw_stream)

    ca.deregister_stream(raw_stream)
    handle.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="uGDS vs GDS pipeline benchmark")
    parser.add_argument("--backend", choices=["ugds", "gds"], required=True)
    parser.add_argument(
        "--gds-file",
        default="/mnt/ugds_test/bench_slab.bin",
        help="File path for GDS slab (only used with --backend gds)",
    )
    args = parser.parse_args()

    print(f"=== Pipeline Benchmark: {args.backend.upper()} ===")
    print(f"IO sizes: {[s // 1024 for s in SIZES]} KB")
    print(f"Depths: {DEPTHS}")
    print(f"Iterations: {ITERS} (warmup: {WARMUP})")
    print()

    if args.backend == "ugds":
        results = run_ugds()
    else:
        results = run_gds(args.gds_file)

    out = f"bench_results_{args.backend}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
