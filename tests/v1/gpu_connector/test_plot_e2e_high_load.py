# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from tests.v1.gpu_connector.plot_e2e_high_load import (
    plot_results,
    validate_pair,
)


def _result(backend: str, scale: float = 1.0) -> dict:
    config = {
        "l1_size_gb": 40,
        "max_model_len": 40960,
        "max_num_seqs": 256,
        "seq_token_counts": [3840, 8192],
        "seq_hot_repeats": 5,
        "conc_token_count": 1024,
        "conc_levels": [1, 2],
        "conc_rounds": 3,
    }
    return {
        "backend": backend,
        "model": "/tmp/Qwen3-0.6B",
        "ssd_only": True,
        "config": config,
        "sequential": [
            {"tokens": 3840, "hot_ttft_p50_ms": 100 * scale},
            {"tokens": 8192, "hot_ttft_p50_ms": 180 * scale},
        ],
        "concurrent": [
            {"concurrency": 1, "median_throughput_ktok_s": 20 / scale},
            {"concurrency": 2, "median_throughput_ktok_s": 30 / scale},
        ],
    }


def test_validate_pair_orders_backends() -> None:
    """Input order does not affect uGDS/GDS assignment."""
    ugds, gds = validate_pair(_result("cufile", 1.2), _result("ugds"))

    assert ugds["backend"] == "ugds"
    assert gds["backend"] == "cufile"


def test_validate_pair_rejects_config_mismatch() -> None:
    """Plots cannot silently compare different experiment matrices."""
    gds = _result("cufile")
    gds["config"]["max_num_seqs"] = 128

    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_pair(_result("ugds"), gds)


def test_plot_results_writes_png(tmp_path: Path) -> None:
    """The high-load comparison output is a non-empty PNG."""
    output = tmp_path / "high-load.png"

    plot_results(_result("ugds"), _result("cufile", 1.2), output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
