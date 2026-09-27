from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


EXPERIMENT_FILES = (
    ("No checkpoint", "large_b1_s2048_none.json"),
    ("k = 1", "large_b1_s2048_group1.json"),
    ("k = 2", "large_b1_s2048_group2.json"),
    ("k = 3", "large_b1_s2048_group3.json"),
)


def _load_results(input_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    return [(label, json.loads((input_dir / filename).read_text())) for label, filename in EXPERIMENT_FILES]


def _label_bars(axis: Any, bars: Any, values: list[float], suffix: str) -> None:
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_results(input_dir: Path, output_svg: Path) -> None:
    results = _load_results(input_dir)
    labels = [label for label, _ in results]
    peak_rss_gib = [result["memory"]["peak_rss_after_backward"] / 1024**3 for _, result in results]
    saved_storage_gib = [result["saved_tensors_after_forward"]["unique_non_parameter_storage_bytes"] / 1024**3 for _, result in results]
    total_seconds = [result["timings_seconds"]["total"] for _, result in results]
    colors = ["#475569", "#16a34a", "#0284c7", "#f97316"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    figure.suptitle(
        "Large Transformer: non-nested activation checkpointing",
        fontsize=15,
        fontweight="bold",
    )

    peak_bars = axes[0].bar(labels, peak_rss_gib, color=colors)
    axes[0].set_title("Process peak RSS")
    axes[0].set_ylabel("GiB")
    axes[0].set_ylim(0, max(peak_rss_gib) * 1.16)
    _label_bars(axes[0], peak_bars, peak_rss_gib, "")

    axes[1].plot(
        labels,
        saved_storage_gib,
        color="#7c3aed",
        marker="o",
        linewidth=2,
        markersize=7,
    )
    axes[1].set_title("Saved unique non-parameter storage")
    axes[1].set_ylabel("GiB, logarithmic")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(saved_storage_gib):
        axes[1].annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    runtime_bars = axes[2].bar(labels, total_seconds, color=colors)
    axes[2].set_title("Forward + backward runtime")
    axes[2].set_ylabel("Seconds")
    axes[2].set_ylim(0, max(total_seconds) * 1.16)
    _label_bars(axes[2], runtime_bars, total_seconds, " s")

    for axis in axes:
        axis.tick_params(axis="x", rotation=22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.text(
        0.5,
        0.015,
        "large, 36 distinct blocks, B=1, S=2048, D=1280, FP32, CPU",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_svg, format="svg")
    plt.close(figure)
    cleaned_svg = "\n".join(line.rstrip() for line in output_svg.read_text().splitlines())
    output_svg.write_text(cleaned_svg + "\n")
    print(f"output_svg={output_svg}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the gradient-checkpointing experiment.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmark_results/gradient_checkpointing"),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("notes/assets/gradient_checkpointing/large_checkpoint_group_comparison.svg"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plot_results(args.input_dir, args.output_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
