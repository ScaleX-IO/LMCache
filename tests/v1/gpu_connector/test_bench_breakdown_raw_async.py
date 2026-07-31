# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the raw-async benchmark helpers."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from tests.v1.gpu_connector.bench_breakdown_raw_async import (
    generate_offsets,
    parse_size,
    percentile,
    read_free_hugepages,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    (("4KiB", 4096), ("1MiB", 1 << 20), ("2GiB", 2 << 30), ("8192", 8192)),
)
def test_parse_size(value: str, expected: int) -> None:
    """Size parsing accepts the units used by the experiment CLI."""
    assert parse_size(value) == expected


def test_parse_size_rejects_unaligned_values() -> None:
    """The direct-I/O workload rejects non-4-KiB-aligned values."""
    with pytest.raises(ValueError, match="4 KiB"):
        parse_size("4097")


def test_generate_offsets_is_deterministic_and_in_range() -> None:
    """Offset generation is reproducible, aligned, and inside the workset."""
    workset = 64 << 20
    io_size = 12 << 20
    first = generate_offsets(workset, io_size, 100, seed=7)
    second = generate_offsets(workset, io_size, 100, seed=7)

    assert first == second
    assert all(offset % 4096 == 0 for offset in first)
    assert all(0 <= offset <= workset - io_size for offset in first)


def test_percentile_uses_nearest_rank() -> None:
    """Percentiles use the documented nearest-rank definition."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.50) == 2.0
    assert percentile(values, 0.95) == 4.0


def test_read_free_hugepages_handles_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing hugepage telemetry is represented explicitly rather than guessed."""
    monkeypatch.setattr(
        "tests.v1.gpu_connector.bench_breakdown_raw_async.NODE0_FREE_HUGEPAGES",
        Path("/definitely/missing/free_hugepages"),
    )
    assert read_free_hugepages() is None
