# SPDX-License-Identifier: Apache-2.0
"""Benchmark compute-free raw async reads for uGDS and cuFile GDS."""

# Standard
from argparse import ArgumentParser, Namespace
from array import array
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import time

# Third Party
import torch

ALIGNMENT = 4096
DEFAULT_SIZES = (4 << 10, 1 << 20, 12 << 20, 16 << 20)
GIB = 1 << 30
NODE0_FREE_HUGEPAGES = Path(
    "/sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages"
)


def parse_size(value: str) -> int:
    """Parse a positive byte count with an optional KiB, MiB, or GiB suffix."""
    normalized = value.strip().lower()
    multipliers = {
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "k": 1 << 10,
        "m": 1 << 20,
        "g": 1 << 30,
    }
    for suffix, multiplier in multipliers.items():
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            break
    else:
        number = normalized
        multiplier = 1
    size = int(number) * multiplier
    if size <= 0 or size % ALIGNMENT:
        raise ValueError(f"size must be a positive 4 KiB multiple: {value}")
    return size


def generate_offsets(workset_bytes: int, io_size: int, count: int, seed: int) -> list[int]:
    """Generate deterministic, aligned offsets that stay inside the workset."""
    if workset_bytes < io_size:
        raise ValueError("workset must be at least as large as the I/O size")
    slots = (workset_bytes - io_size) // ALIGNMENT + 1
    generator = random.Random(seed + io_size)
    return [generator.randrange(slots) * ALIGNMENT for _ in range(count)]


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile from an already sorted sample."""
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    index = math.ceil(fraction * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
    temporary.replace(path)


def save_offsets(path: Path, offsets: list[int]) -> str:
    """Save an offset sequence and return its SHA-256 digest."""
    values = array("Q", offsets)
    with path.open("wb") as output:
        values.tofile(output)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_backend(backend: str) -> ModuleType:
    """Load the selected low-level async backend module."""
    module_name = "_ugds_async" if backend == "ugds" else "_cufile_async"
    return importlib.import_module(f"lmcache.v1.gpu_connector.{module_name}")


def read_free_hugepages() -> int | None:
    """Read the NUMA-node-0 free 2-MiB hugepage count when available."""
    try:
        return int(NODE0_FREE_HUGEPAGES.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def open_handle(
    backend: str,
    module: ModuleType,
    path: str,
    workset_bytes: int,
    initialize: bool,
) -> Any:
    """Open and register one raw uGDS device or O_DIRECT cuFile slab."""
    if backend == "cufile" and initialize:
        create_fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.posix_fallocate(create_fd, 0, workset_bytes)
        finally:
            os.close(create_fd)
    if backend == "cufile" and os.path.getsize(path) < workset_bytes:
        raise ValueError(f"cuFile slab is smaller than workset: {path}")

    flags = os.O_RDWR if backend == "ugds" else os.O_RDWR | os.O_DIRECT
    file_descriptor = os.open(path, flags)
    try:
        raw_handle = module.register_handle(file_descriptor)
    except Exception:
        os.close(file_descriptor)
        raise
    return module.AsyncHandle.from_fd(
        file_descriptor,
        raw_handle,
        path,
        writable=True,
    )


def transfer_one(
    handle: Any,
    operation: str,
    buffer: torch.Tensor,
    size: int,
    offset: int,
    raw_stream: int,
) -> None:
    """Submit one async operation, synchronize it, and verify byte count."""
    transfer = handle.write_async if operation == "write" else handle.read_async
    submission = transfer(
        buffer.data_ptr(),
        size,
        file_offset=offset,
        buf_offset=0,
        raw_stream=raw_stream,
    )
    torch.cuda.synchronize()
    if submission.bytes_done != size:
        raise RuntimeError(
            f"short {operation} at offset {offset}: "
            f"expected {size}, got {submission.bytes_done}"
        )


def initialize_workset(
    handle: Any,
    buffer: torch.Tensor,
    workset_bytes: int,
    raw_stream: int,
) -> None:
    """Initialize the complete workset with zeroes through the tested backend."""
    buffer.zero_()
    torch.cuda.synchronize()
    chunk_bytes = buffer.numel()
    started = time.perf_counter()
    for offset in range(0, workset_bytes, chunk_bytes):
        size = min(chunk_bytes, workset_bytes - offset)
        transfer_one(handle, "write", buffer, size, offset, raw_stream)
    elapsed = time.perf_counter() - started
    bandwidth = workset_bytes / elapsed / GIB
    print(f"initialized {workset_bytes / GIB:.1f} GiB in {elapsed:.2f}s ({bandwidth:.2f} GiB/s)")


def validate_workset(
    handle: Any,
    buffer: torch.Tensor,
    workset_bytes: int,
    raw_stream: int,
) -> None:
    """Check zero data at the beginning, middle, and end of the workset."""
    size = min(buffer.numel(), 1 << 20)
    offsets = (0, (workset_bytes // 2) // ALIGNMENT * ALIGNMENT, workset_bytes - size)
    for offset in offsets:
        buffer[:size].fill_(0xFF)
        torch.cuda.synchronize()
        transfer_one(handle, "read", buffer, size, offset, raw_stream)
        mismatches = torch.count_nonzero(buffer[:size]).item()
        if mismatches:
            raise RuntimeError(
                f"data validation failed at offset {offset}: {mismatches} non-zero bytes"
            )
    print("data validation: PASS")


def benchmark_size(
    handle: Any,
    buffer: torch.Tensor,
    io_size: int,
    warmup_offsets: list[int],
    measured_offsets: list[int],
    raw_stream: int,
) -> dict[str, Any]:
    """Measure one I/O size at physical queue depth one."""
    for offset in warmup_offsets:
        transfer_one(handle, "read", buffer, io_size, offset, raw_stream)

    latencies_us: list[float] = []
    wall_started = time.perf_counter_ns()
    for offset in measured_offsets:
        started = time.perf_counter_ns()
        transfer_one(handle, "read", buffer, io_size, offset, raw_stream)
        latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
    wall_ns = time.perf_counter_ns() - wall_started

    sorted_latencies = sorted(latencies_us)
    transferred_bytes = io_size * len(measured_offsets)
    bandwidth_gib_s = transferred_bytes / (wall_ns / 1e9) / GIB
    return {
        "io_size": io_size,
        "operations": len(measured_offsets),
        "bytes": transferred_bytes,
        "wall_ns": wall_ns,
        "bandwidth_gib_s": bandwidth_gib_s,
        "latency_us": {
            "p50": percentile(sorted_latencies, 0.50),
            "p95": percentile(sorted_latencies, 0.95),
            "p99": percentile(sorted_latencies, 0.99),
            "mean": statistics.fmean(sorted_latencies),
        },
        "raw_latency_us": latencies_us,
    }


def run(args: Namespace) -> Path:
    """Execute one raw-async backend round and return its result path."""
    print(f"raw-async round {args.round}: backend={args.backend}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(args.gpu)
    print(f"CUDA device {args.gpu}: {torch.cuda.get_device_name(args.gpu)}", flush=True)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    sizes = tuple(parse_size(value) for value in args.sizes.split(","))
    if max(sizes) > args.buffer_bytes:
        raise ValueError("buffer must be at least as large as the maximum I/O size")

    hugepages_before_handle = read_free_hugepages()
    module = load_backend(args.backend)
    print(f"loaded backend module: {module.__name__}", flush=True)
    handle = open_handle(
        args.backend,
        module,
        args.path,
        args.workset_bytes,
        args.initialize,
    )
    hugepages_after_handle = read_free_hugepages()
    print(f"registered backend handle: {args.path}", flush=True)
    buffer = torch.empty(args.buffer_bytes, dtype=torch.uint8, device="cuda")
    print(f"allocated GPU buffer: {args.buffer_bytes} bytes", flush=True)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    module.register_buffer(buffer)
    print("registered GPU buffer", flush=True)
    module.register_stream(raw_stream)
    print("registered CUDA stream", flush=True)

    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "scenario": "raw_async",
        "backend": args.backend,
        "round": args.round,
        "started_at_utc": started_at.isoformat(),
        "path": args.path,
        "gpu": args.gpu,
        "workset_bytes": args.workset_bytes,
        "seed": args.seed,
        "warmup_operations": args.warmup,
        "minimum_operations": args.minimum_operations,
        "minimum_bytes": args.minimum_bytes,
        "initialized": args.initialize,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "node0_free_hugepages_before_handle": hugepages_before_handle,
        "node0_free_hugepages_after_handle": hugepages_after_handle,
        "measurements": [],
    }

    try:
        if args.initialize:
            initialize_workset(handle, buffer, args.workset_bytes, raw_stream)
        validate_workset(handle, buffer, args.workset_bytes, raw_stream)

        for io_size in sizes:
            operations = max(
                args.minimum_operations,
                math.ceil(args.minimum_bytes / io_size),
            )
            offsets = generate_offsets(args.workset_bytes, io_size, operations, args.seed)
            warmup_offsets = generate_offsets(
                args.workset_bytes,
                io_size,
                args.warmup,
                args.seed ^ 0xA5A5A5A5,
            )
            offset_path = output_directory / (
                f"offsets_seed{args.seed}_size{io_size}_count{operations}.u64"
            )
            if offset_path.exists():
                existing = array("Q")
                with offset_path.open("rb") as offset_input:
                    existing.fromfile(offset_input, operations)
                if existing.tolist() != offsets:
                    raise RuntimeError(f"existing offset sequence differs: {offset_path}")
                offset_digest = hashlib.sha256(offset_path.read_bytes()).hexdigest()
            else:
                offset_digest = save_offsets(offset_path, offsets)

            measurement = benchmark_size(
                handle,
                buffer,
                io_size,
                warmup_offsets,
                offsets,
                raw_stream,
            )
            measurement["offset_file"] = offset_path.name
            measurement["offset_sha256"] = offset_digest
            result["measurements"].append(measurement)
            print(
                f"size={io_size:>8d} ops={operations:>7d} "
                f"bw={measurement['bandwidth_gib_s']:.3f} GiB/s "
                f"p50={measurement['latency_us']['p50']:.1f} us"
            )
    finally:
        module.deregister_stream(raw_stream)
        module.deregister_buffer(buffer)
        handle.close()

    result["node0_free_hugepages_after_close"] = read_free_hugepages()
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result_path = output_directory / (
        f"raw_async_{args.backend}_round{args.round}.json"
    )
    write_json(result_path, result)
    return result_path


def parse_args() -> Namespace:
    """Parse command-line arguments for one raw-async round."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ugds", "cufile"), required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output-dir", default="results/breakdown/runs/raw_async")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--sizes", default="4KiB,1MiB,12MiB,16MiB")
    parser.add_argument("--workset-bytes", type=parse_size, default=parse_size("40GiB"))
    parser.add_argument("--buffer-bytes", type=parse_size, default=parse_size("16MiB"))
    parser.add_argument("--minimum-bytes", type=parse_size, default=parse_size("2GiB"))
    parser.add_argument("--minimum-operations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--initialize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    print(f"result: {artifact}", flush=True)
    if arguments.backend == "cufile":
        os._exit(0)
