# SPDX-License-Identifier: Apache-2.0
"""Plot uGDS and GDS high-load E2E results on one PNG canvas."""

# Standard
import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lmcache-matplotlib")

# Third Party
import matplotlib.pyplot as plt


def load_result(path: Path) -> dict[str, Any]:
    """Load and minimally validate one high-load result file.

    Args:
        path: Benchmark JSON path.

    Returns:
        Parsed result dictionary.

    Raises:
        ValueError: If the file is not an SSD-only result or lacks required data.
    """
    result = json.loads(path.read_text())
    if result.get("ssd_only") is not True:
        raise ValueError(f"{path} is not an --ssd-only result")
    if not result.get("sequential") or not result.get("concurrent"):
        raise ValueError(f"{path} lacks sequential or concurrent results")
    return result


def validate_pair(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and order a uGDS/cuFile result pair.

    Args:
        first: First parsed benchmark result.
        second: Second parsed benchmark result.

    Returns:
        Results ordered as ``(ugds, gds)``.

    Raises:
        ValueError: If backends, model, or fixed experiment parameters differ.
    """
    by_backend = {first.get("backend"): first, second.get("backend"): second}
    if set(by_backend) != {"ugds", "cufile"}:
        raise ValueError("inputs must contain one ugds and one cufile result")
    ugds = by_backend["ugds"]
    gds = by_backend["cufile"]
    if ugds.get("model") != gds.get("model"):
        raise ValueError("model mismatch between benchmark results")
    fixed_keys = (
        "l1_size_gb",
        "max_model_len",
        "max_num_seqs",
        "seq_token_counts",
        "seq_hot_repeats",
        "conc_token_count",
        "conc_levels",
        "conc_rounds",
    )
    for key in fixed_keys:
        if ugds["config"].get(key) != gds["config"].get(key):
            raise ValueError(f"configuration mismatch for {key}")
    ugds_contexts = [sample["tokens"] for sample in ugds["sequential"]]
    gds_contexts = [sample["tokens"] for sample in gds["sequential"]]
    if ugds_contexts != gds_contexts:
        raise ValueError("sequential result points do not match")
    return ugds, gds


def plot_results(
    ugds: dict[str, Any], gds: dict[str, Any], output: Path
) -> None:
    """Create the two-panel TTFT and throughput comparison PNG.

    Args:
        ugds: Validated uGDS result.
        gds: Validated cuFile GDS result.
        output: Destination PNG path.

    Raises:
        ValueError: If output is not PNG or no common concurrency points exist.
    """
    if output.suffix.lower() != ".png":
        raise ValueError("high-load plot output must use .png")

    context_tokens = [sample["tokens"] for sample in ugds["sequential"]]
    context_positions = list(range(len(context_tokens)))
    ugds_ttft = [sample["hot_ttft_p50_ms"] for sample in ugds["sequential"]]
    gds_ttft = [sample["hot_ttft_p50_ms"] for sample in gds["sequential"]]

    ugds_concurrent = {
        sample["concurrency"]: sample["median_throughput_ktok_s"]
        for sample in ugds["concurrent"]
    }
    gds_concurrent = {
        sample["concurrency"]: sample["median_throughput_ktok_s"]
        for sample in gds["concurrent"]
    }
    concurrency = sorted(set(ugds_concurrent) & set(gds_concurrent))
    if not concurrency:
        raise ValueError("no common successful concurrency points")
    ugds_throughput = [ugds_concurrent[level] for level in concurrency]
    gds_throughput = [gds_concurrent[level] for level in concurrency]

    figure, (ttft_axis, throughput_axis) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"ugds": "#1E88E5", "gds": "#FB8C00"}

    ttft_axis.plot(
        context_positions,
        ugds_ttft,
        marker="o",
        linewidth=2,
        color=colors["ugds"],
        label="uGDS",
    )
    ttft_axis.plot(
        context_positions,
        gds_ttft,
        marker="s",
        linewidth=2,
        color=colors["gds"],
        label="cuFile GDS",
    )
    for position, ugds_value, gds_value in zip(
        context_positions, ugds_ttft, gds_ttft, strict=True
    ):
        ttft_axis.vlines(
            position,
            min(ugds_value, gds_value),
            max(ugds_value, gds_value),
            colors="#666666",
            linestyles=":",
            linewidth=1,
        )
        ttft_axis.annotate(
            f"{gds_value / ugds_value:.2f}×",
            (position, (ugds_value + gds_value) / 2),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )
    ttft_axis.set_xticks(
        context_positions, [f"{value:,}" for value in context_tokens]
    )
    ttft_axis.set_xlabel("Prompt tokens")
    ttft_axis.set_ylabel("Hot TTFT p50 (ms, lower is better)")
    ttft_axis.set_title("Context pressure: GDS / uGDS ratio")
    ttft_axis.grid(True, alpha=0.25)
    ttft_axis.legend()

    throughput_axis.plot(
        concurrency,
        ugds_throughput,
        marker="o",
        linewidth=2,
        color=colors["ugds"],
        label="uGDS",
    )
    throughput_axis.plot(
        concurrency,
        gds_throughput,
        marker="s",
        linewidth=2,
        color=colors["gds"],
        label="cuFile GDS",
    )
    for level, ugds_value, gds_value in zip(
        concurrency, ugds_throughput, gds_throughput, strict=True
    ):
        throughput_axis.vlines(
            level,
            min(ugds_value, gds_value),
            max(ugds_value, gds_value),
            colors="#666666",
            linestyles=":",
            linewidth=1,
        )
        throughput_axis.annotate(
            f"{ugds_value / gds_value:.2f}×",
            (level, (ugds_value + gds_value) / 2),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )
    throughput_axis.set_xscale("log", base=2)
    throughput_axis.set_xticks(concurrency, [str(value) for value in concurrency])
    throughput_axis.set_xlabel("Concurrent requests (1,024 tokens each)")
    throughput_axis.set_ylabel("Throughput (kTokens/s, higher is better)")
    throughput_axis.set_title("Concurrency pressure: uGDS / GDS ratio")
    throughput_axis.grid(True, alpha=0.25)
    throughput_axis.legend()

    figure.suptitle("LMCache SSD-only E2E: Qwen3-0.6B on A100", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Parse result paths and write the validated comparison PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ugds_json", type=Path)
    parser.add_argument("gds_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ugds, gds = validate_pair(
        load_result(args.ugds_json), load_result(args.gds_json)
    )
    plot_results(ugds, gds, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
