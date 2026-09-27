"""Plot CPU attention benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


COLORS = {
    16: "#0072B2",
    32: "#009E73",
    64: "#E69F00",
    128: "#D55E00",
}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload["cases"])


def _saved_storage_bytes(batch_size: int, sequence_length: int, d_model: int) -> int:
    activation = batch_size * sequence_length * d_model * 4
    score = batch_size * sequence_length**2 * 4
    row_statistics = batch_size * sequence_length * 12
    return 2 * score + 3 * activation + row_statistics


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_results(input_json: Path, output_svg: Path) -> None:
    cases = _load_cases(input_json)
    successful = [case for case in cases if case["status"] == "ok"]
    batch_size = int(cases[0]["config"]["batch_size"])
    sequence_lengths = sorted({int(case["config"]["sequence_length"]) for case in cases})
    d_models = sorted({int(case["config"]["d_model"]) for case in cases})

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "cs336-attention-cpu",
        }
    )
    figure, (timing_axis, memory_axis) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    for d_model in d_models:
        model_cases = [case for case in successful if int(case["config"]["d_model"]) == d_model]
        x_values = [int(case["config"]["sequence_length"]) for case in model_cases]
        color = COLORS[d_model]
        for phase, marker, line_style in (
            ("forward", "o", "-"),
            ("backward", "s", "--"),
        ):
            y_values = [float(case["timings"][phase]["mean_ms"]) for case in model_cases]
            timing_axis.plot(
                x_values,
                y_values,
                color=color,
                linestyle=line_style,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                label=f"d={d_model}, {phase}",
            )

        measured_gib = [float(case["saved_tensors_after_forward"]["unique_storage_bytes"]) / 1024**3 for case in model_cases]
        memory_axis.plot(
            x_values,
            measured_gib,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.5,
            label=f"d={d_model} measured",
        )

    theoretical_gib = [_saved_storage_bytes(batch_size, sequence_length, d_models[0]) / 1024**3 for sequence_length in sequence_lengths]
    memory_axis.plot(
        sequence_lengths,
        theoretical_gib,
        color="#333333",
        linestyle=":",
        linewidth=2,
        label="d=16 theory (incl. OOM)",
    )

    for axis in (timing_axis, memory_axis):
        axis.axvspan(8192, 16384, color="#D55E00", alpha=0.08)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(
            sequence_lengths,
            labels=[str(value) for value in sequence_lengths],
        )
        axis.set_xlabel("Sequence length S")
        _style_axis(axis)

    timing_axis.set_title(
        "(a) Mean latency, 100 measurements",
        loc="left",
        fontweight="bold",
    )
    timing_axis.set_ylabel("Milliseconds per pass")
    timing_axis.legend(ncols=2, frameon=False, fontsize=7.5)

    memory_axis.set_title(
        "(b) Autograd saved storage after forward",
        loc="left",
        fontweight="bold",
    )
    memory_axis.set_ylabel("Unique storage (GiB)")
    memory_axis.legend(frameon=False, fontsize=7.5)
    memory_axis.annotate(
        "OOM under 20 GiB RLIMIT_AS",
        xy=(8192, theoretical_gib[-2]),
        xytext=(5000, 10),
        arrowprops={"arrowstyle": "->", "color": "#D55E00"},
        color="#A33B00",
        fontsize=8.5,
    )

    figure.suptitle(
        "Naive scaled-dot-product attention on CPU (B=8, FP32, 50 threads)",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_svg,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(figure)
    print(f"output_svg={output_svg}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the CPU attention benchmark.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("benchmark_results/cpu_attention/sweep.json"),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("notes/assets/attention/cpu_attention_benchmark.svg"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plot_results(args.input_json, args.output_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
