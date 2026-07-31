# SPDX-License-Identifier: Apache-2.0
"""Benchmark the production-shaped LMCache GDS transaction."""

# Standard
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.gpu_connector.gds_context import GDSContext, SlabDirection
from lmcache.v1.memory_management import GDSMemoryObject
from tests.v1.gpu_connector.bench_breakdown_raw_async import (
    GIB,
    load_backend,
    open_handle,
    parse_size,
    percentile,
    read_free_hugepages,
    write_json,
)

CHUNK_BYTES = 28 << 20
FIRST_SEGMENT_BYTES = 16 << 20
CHUNKS_PER_TRANSACTION = 4
TRANSACTION_BYTES = CHUNK_BYTES * CHUNKS_PER_TRANSACTION


def parse_depths(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly increasing transaction-depth list."""
    depths = tuple(int(item.strip()) for item in value.split(","))
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("depths must contain positive integers")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("depths must be unique and strictly increasing")
    return depths


def segment_plan(transaction_depth: int) -> list[tuple[int, int, int]]:
    """Return ``(buffer index, slab offset, size)`` for production-shaped I/O."""
    plan: list[tuple[int, int, int]] = []
    for buffer_index in range(transaction_depth * CHUNKS_PER_TRANSACTION):
        chunk_offset = buffer_index * CHUNK_BYTES
        plan.append((buffer_index, chunk_offset, FIRST_SEGMENT_BYTES))
        plan.append(
            (
                buffer_index,
                chunk_offset + FIRST_SEGMENT_BYTES,
                CHUNK_BYTES - FIRST_SEGMENT_BYTES,
            )
        )
    return plan


def offset_digest(transaction_depth: int) -> str:
    """Hash the ordered logical segment sequence for cross-mode validation."""
    encoded = json.dumps(segment_plan(transaction_depth), separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def summarize_samples(samples: list[float]) -> dict[str, float]:
    """Summarize a non-empty latency sample using nearest-rank percentiles."""
    ordered = sorted(samples)
    return {
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "mean": statistics.fmean(ordered),
    }


def check_submissions(submissions: list[Any], expected_sizes: list[int]) -> None:
    """Raise when an async backend reports a short read."""
    if len(submissions) != len(expected_sizes):
        raise RuntimeError("submission count does not match the segment plan")
    for index, (submission, expected) in enumerate(zip(submissions, expected_sizes)):
        if submission.bytes_done != expected:
            raise RuntimeError(
                f"short read for segment {index}: expected {expected}, "
                f"got {submission.bytes_done}"
            )


def benchmark_depth(
    depth: int,
    warmup: int,
    iterations: int,
    submit: Callable[[int], list[Any]],
    synchronize: Callable[[], None],
    expected_sizes: list[int] | None,
) -> dict[str, Any]:
    """Measure one transaction depth with one synchronization per submitted batch."""
    for _ in range(warmup):
        submissions = submit(depth)
        synchronize()
        if expected_sizes is not None:
            check_submissions(submissions, expected_sizes[: len(submissions)])

    submit_us: list[float] = []
    wall_us: list[float] = []
    cpu_us: list[float] = []
    for _ in range(iterations):
        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        submit_started = time.perf_counter_ns()
        submissions = submit(depth)
        submit_finished = time.perf_counter_ns()
        synchronize()
        wall_finished = time.perf_counter_ns()
        cpu_finished = time.process_time_ns()
        if expected_sizes is not None:
            check_submissions(submissions, expected_sizes[: len(submissions)])
        submit_us.append((submit_finished - submit_started) / 1000.0)
        wall_us.append((wall_finished - wall_started) / 1000.0)
        cpu_us.append((cpu_finished - cpu_started) / 1000.0)

    wall_seconds = sum(wall_us) / 1e6
    logical_bytes = depth * TRANSACTION_BYTES * iterations
    return {
        "transaction_depth": depth,
        "transactions_per_iteration": depth,
        "chunks_per_transaction": CHUNKS_PER_TRANSACTION,
        "segments_per_transaction": CHUNKS_PER_TRANSACTION * 2,
        "bytes_per_transaction": TRANSACTION_BYTES,
        "iterations": iterations,
        "logical_bytes": logical_bytes,
        "bandwidth_gib_s": logical_bytes / wall_seconds / GIB,
        "batch_wall_us": summarize_samples(wall_us),
        "submit_us": summarize_samples(submit_us),
        "cpu_us": summarize_samples(cpu_us),
        "amortized_transaction_us_p50": percentile(sorted(wall_us), 0.50) / depth,
        "worker_cpu_utilization": sum(cpu_us) / sum(wall_us),
        "raw_batch_wall_us": wall_us,
        "raw_submit_us": submit_us,
        "raw_cpu_us": cpu_us,
        "offset_sha256": offset_digest(depth),
    }


def validate_zero_reads(
    buffers: list[torch.Tensor],
    depth: int,
    submit: Callable[[int], list[Any]],
    synchronize: Callable[[], None],
    expected_sizes: list[int] | None,
) -> None:
    """Overwrite destinations, read the slab, and verify every transferred byte."""
    for buffer in buffers[: depth * CHUNKS_PER_TRANSACTION]:
        buffer.fill_(0xFF)
    synchronize()
    submissions = submit(depth)
    synchronize()
    if expected_sizes is not None:
        check_submissions(submissions, expected_sizes[: len(submissions)])
    mismatch_counts = [
        torch.count_nonzero(buffer)
        for buffer in buffers[: depth * CHUNKS_PER_TRANSACTION]
    ]
    mismatches = torch.stack(mismatch_counts).sum().item()
    if mismatches:
        raise RuntimeError(f"data validation failed: {mismatches} non-zero bytes")


def allocate_buffers(max_depth: int) -> list[torch.Tensor]:
    """Allocate one independent 28 MiB staging slot per LMCache chunk."""
    return [
        torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="cuda")
        for _ in range(max_depth * CHUNKS_PER_TRANSACTION)
    ]


def run_raw(
    args: Namespace,
    buffers: list[torch.Tensor],
    depths: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Run the transaction directly against the selected async backend."""
    module = load_backend(args.backend)
    handle = open_handle(args.backend, module, args.path, args.workset_bytes, False)
    stream = torch.cuda.current_stream()
    raw_stream = stream.cuda_stream
    registered_regions: list[torch.Tensor] = []
    module.register_stream(raw_stream)
    for buffer in buffers:
        regions = (buffer[:FIRST_SEGMENT_BYTES], buffer[FIRST_SEGMENT_BYTES:])
        for region in regions:
            module.register_buffer(region)
            registered_regions.append(region)

    plans = {depth: segment_plan(depth) for depth in depths}

    def submit(depth: int) -> list[Any]:
        submissions: list[Any] = []
        for buffer_index, slab_offset, size in plans[depth]:
            region_start = 0 if size == FIRST_SEGMENT_BYTES else FIRST_SEGMENT_BYTES
            region = buffers[buffer_index][region_start : region_start + size]
            submissions.append(
                handle.read_async(
                    region.data_ptr(),
                    size,
                    file_offset=slab_offset,
                    buf_offset=0,
                    raw_stream=raw_stream,
                )
            )
        return submissions

    expected_sizes = [size for _, _, size in plans[max(depths)]]
    try:
        validate_zero_reads(
            buffers, max(depths), submit, torch.cuda.synchronize, expected_sizes
        )
        print("data validation: PASS", flush=True)
        return [
            benchmark_depth(
                depth,
                args.warmup,
                args.iterations,
                submit,
                torch.cuda.synchronize,
                expected_sizes,
            )
            for depth in depths
        ]
    finally:
        module.deregister_stream(raw_stream)
        for region in registered_regions:
            module.deregister_buffer(region)
        handle.close()


def run_context(
    args: Namespace,
    buffers: list[torch.Tensor],
    depths: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Run the transaction through the production GDSContext path."""
    config = GdsL1Config(
        file_location=args.path,
        size_in_bytes=args.workset_bytes,
        backend=args.backend,
    )
    context = GDSContext()
    context.initialize(config)
    for buffer in buffers:
        context.register_gpu_buffer(buffer)

    manager = GDSL1MemoryManager(config)
    error, allocated = manager.allocate(
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK_BYTES])], dtypes=[torch.uint8]),
        len(buffers),
    )
    if error != L1Error.SUCCESS:
        context.close()
        raise RuntimeError(f"failed to allocate transaction slab objects: {error}")
    memory_objects = [obj for obj in allocated if isinstance(obj, GDSMemoryObject)]
    if len(memory_objects) != len(buffers):
        context.close()
        raise RuntimeError("GDS memory manager returned an unexpected object type")

    for buffer in buffers:
        buffer.zero_()
    torch.cuda.synchronize()
    for memory_object, buffer in zip(memory_objects, buffers):
        context.transfer_async(memory_object, buffer, SlabDirection.WRITE)
    torch.cuda.synchronize()
    initialized_bytes = len(buffers) * CHUNK_BYTES
    print(
        f"initialized {initialized_bytes / GIB:.2f} GiB of context slab data",
        flush=True,
    )

    def submit(depth: int) -> list[Any]:
        for index in range(depth * CHUNKS_PER_TRANSACTION):
            context.transfer_async(
                memory_objects[index], buffers[index], SlabDirection.READ
            )
        return []

    try:
        validate_zero_reads(buffers, max(depths), submit, torch.cuda.synchronize, None)
        print("data validation: PASS", flush=True)
        return [
            benchmark_depth(
                depth,
                args.warmup,
                args.iterations,
                submit,
                torch.cuda.synchronize,
                None,
            )
            for depth in depths
        ]
    finally:
        manager.free(allocated)
        context.close()


def run(args: Namespace) -> Path:
    """Execute one production-transaction round and return its JSON artifact."""
    print(
        f"production-transaction round {args.round}: "
        f"backend={args.backend} mode={args.mode}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(args.gpu)
    depths = parse_depths(args.depths)
    if max(depths) * TRANSACTION_BYTES > args.workset_bytes:
        raise ValueError("the largest transaction batch exceeds the slab workset")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    buffers = allocate_buffers(max(depths))
    print(
        f"allocated {len(buffers)} x 28 MiB GPU staging buffers "
        f"({len(buffers) * CHUNK_BYTES / GIB:.2f} GiB)",
        flush=True,
    )
    started_at = datetime.now(timezone.utc)
    hugepages_before = read_free_hugepages()
    if args.mode == "raw":
        measurements = run_raw(args, buffers, depths)
    else:
        measurements = run_context(args, buffers, depths)
    result = {
        "scenario": "production_transaction",
        "backend": args.backend,
        "mode": args.mode,
        "round": args.round,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": args.path,
        "gpu": args.gpu,
        "workset_bytes": args.workset_bytes,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "depths": list(depths),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "node0_free_hugepages_before": hugepages_before,
        "node0_free_hugepages_after": read_free_hugepages(),
        "measurements": measurements,
    }
    result_path = output_directory / (
        f"production_transaction_{args.backend}_{args.mode}_round{args.round}.json"
    )
    write_json(result_path, result)
    return result_path


def parse_args() -> Namespace:
    """Parse command-line arguments for one production-transaction round."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ugds", "cufile"), required=True)
    parser.add_argument("--mode", choices=("raw", "context"), required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument(
        "--output-dir", default="results/breakdown/runs/production_transaction"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--depths", default="1,8,64")
    parser.add_argument("--workset-bytes", type=parse_size, default=parse_size("40GiB"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    print(f"result: {artifact}", flush=True)
    if arguments.backend == "cufile":
        os._exit(0)
