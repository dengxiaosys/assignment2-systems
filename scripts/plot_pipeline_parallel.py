"""Render the pipeline-parallel microbatch scaling experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"pipeline benchmark did not pass: {summary.get('status')}")
    if not summary.get("aggregate"):
        raise ValueError("pipeline benchmark has no aggregate results")
    return summary


def plot_microbatch_scaling(summary: dict[str, Any], output_path: Path) -> None:
    """Plot measured latency and ideal utilization against microbatch count."""

    aggregate = summary["aggregate"]
    microbatch_counts = sorted(int(value) for value in aggregate)
    step_means = [aggregate[str(value)]["step_mean_ms"] for value in microbatch_counts]
    step_errors = [aggregate[str(value)]["step_repeat_std_ms"] for value in microbatch_counts]
    throughputs = [aggregate[str(value)]["tokens_per_second"] for value in microbatch_counts]
    ideal_efficiencies = [100 * aggregate[str(value)]["ideal_fill_drain_efficiency"] for value in microbatch_counts]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    axes[0].errorbar(
        microbatch_counts,
        step_means,
        yerr=step_errors,
        marker="o",
        capsize=4,
        color="#0072B2",
        linewidth=2,
    )
    axes[0].set_title("Measured end-to-end latency")
    axes[0].set_xlabel("Microbatches per mini-batch")
    axes[0].set_ylabel("Rank-max step time (ms)")

    throughput_bars = axes[1].bar(
        microbatch_counts,
        throughputs,
        width=0.65,
        color="#009E73",
        label="Measured throughput",
    )
    efficiency_axis = axes[1].twinx()
    efficiency_axis.plot(
        microbatch_counts,
        ideal_efficiencies,
        color="#D55E00",
        marker="s",
        linewidth=2,
        label="Ideal pipeline efficiency",
    )
    axes[1].set_title("Measured throughput vs. ideal utilization")
    axes[1].set_xlabel("Microbatches per mini-batch")
    axes[1].set_ylabel("Tokens/s")
    efficiency_axis.set_ylabel("Ideal efficiency (%)")
    efficiency_axis.set_ylim(0, 100)
    axes[1].bar_label(throughput_bars, fmt="%.0f", padding=3, fontsize=9)

    for axis in axes:
        axis.set_xticks(microbatch_counts)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    secondary_handles, secondary_labels = efficiency_axis.get_legend_handles_labels()
    axes[1].legend(handles + secondary_handles, labels + secondary_labels, frameon=False, loc="center right")
    figure.suptitle("Two-stage GPipe on CPU/Gloo: smaller bubbles do not guarantee lower latency")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, metadata={"Date": None})
    plt.close(figure)
    if output_path.suffix == ".svg":
        lines = output_path.read_text(encoding="utf-8").splitlines()
        output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_microbatch_scaling(_load_summary(args.summary), args.output)
    print(f"plot_status=passed output={args.output}")


if __name__ == "__main__":
    main()
