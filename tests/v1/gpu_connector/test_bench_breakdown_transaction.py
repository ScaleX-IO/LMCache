# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the production-transaction benchmark helpers."""

# Third Party
import pytest

# First Party
from tests.v1.gpu_connector.bench_breakdown_transaction import (
    CHUNK_BYTES,
    CHUNKS_PER_TRANSACTION,
    FIRST_SEGMENT_BYTES,
    offset_digest,
    parse_depths,
    segment_plan,
    summarize_samples,
)


def test_parse_depths_accepts_minimal_matrix() -> None:
    """The approved minimal depth matrix parses without reordering."""
    assert parse_depths("1,8,64") == (1, 8, 64)


@pytest.mark.parametrize("value", ("", "0,1", "8,1", "1,1"))
def test_parse_depths_rejects_invalid_sequences(value: str) -> None:
    """Depths must be positive, unique, and strictly increasing."""
    with pytest.raises(ValueError):
        parse_depths(value)


def test_segment_plan_matches_one_production_transaction() -> None:
    """One transaction consists of four adjacent 16+12 MiB chunk segments."""
    plan = segment_plan(1)
    assert len(plan) == CHUNKS_PER_TRANSACTION * 2
    assert plan[0] == (0, 0, FIRST_SEGMENT_BYTES)
    assert plan[1] == (0, FIRST_SEGMENT_BYTES, CHUNK_BYTES - FIRST_SEGMENT_BYTES)
    assert plan[-1][1] + plan[-1][2] == CHUNKS_PER_TRANSACTION * CHUNK_BYTES


def test_offset_digest_is_depth_specific_and_deterministic() -> None:
    """The manifest digest detects changes in the ordered logical I/O plan."""
    assert offset_digest(8) == offset_digest(8)
    assert offset_digest(1) != offset_digest(8)


def test_summarize_samples_uses_nearest_rank() -> None:
    """Batch latency summaries preserve the experiment percentile definition."""
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0])
    assert summary == {"p50": 2.0, "p95": 4.0, "p99": 4.0, "mean": 2.5}
