from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch


INPUT_DIR = Path("benchmark_results/cpu_memory_profile")
OUTPUT_DIR = Path("notes/assets/memory_profiling")
CONTEXT_LENGTHS = (128, 2048)
PRECISION_LABELS = {"fp32": "FP32"}
PRECISION_COLORS = {"fp32": "#0f766e"}
BLOCK_SOURCE_LABELS = {
    "model.py:133:scaled_dot_product_attention": "Attention P@V matmul",
    "nn_utils.py:11:softmax": "Softmax division",
    "nn_utils.py:10:softmax": "Softmax exp/subtract",
    "model.py:77:silu": "SiLU",
    "model.py:32:forward": "Linear matmul",
}
STAGE_COLORS = {
    "baseline": "#f1f5f9",
    "zero_grad": "#dbeafe",
    "forward": "#ccfbf1",
    "loss": "#fef3c7",
    "backward": "#e9d5ff",
    "optimizer": "#fed7aa",
    "complete": "#f1f5f9",
}


def _load(context_length: int, mode: str, precision: str) -> dict[str, Any]:
    path = INPUT_DIR / f"large_s{context_length}_{mode}_{precision}.json"
    return json.loads(path.read_text())


def _load_steps(context_length: int, mode: str, precision: str, steps: int) -> dict[str, Any]:
    suffix = "" if steps == 1 else f"_{steps}steps"
    path = INPUT_DIR / f"large_s{context_length}_{mode}_{precision}{suffix}.json"
    return json.loads(path.read_text())


def _clean_svg(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def _plot_timeline(axis: Axes, result: dict[str, Any], precision: str) -> None:
    samples = result["samples"]
    times_seconds = np.asarray([sample["elapsed_ms"] / 1_000 for sample in samples])
    rss_gib = np.asarray([sample["rss_bytes"] / 1024**3 for sample in samples])
    stages = [sample["stage"] for sample in samples]
    steps = [int(sample.get("step", 1)) for sample in samples]

    segment_start = 0
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[segment_start]:
            left = times_seconds[segment_start]
            right = times_seconds[index - 1]
            stage = stages[segment_start]
            axis.axvspan(left, max(right, left + 1e-6), color=STAGE_COLORS[stage], alpha=0.45, linewidth=0)
            if steps[segment_start] <= 1 and right - left > 0.06 * max(times_seconds[-1], 1e-9):
                axis.text(
                    (left + right) / 2,
                    0.98,
                    stage,
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#475569",
                )
            segment_start = index

    color = PRECISION_COLORS[precision]
    axis.plot(times_seconds, rss_gib, color=color, linewidth=1.6)
    peak_index = int(np.argmax(rss_gib))
    axis.scatter(times_seconds[peak_index], rss_gib[peak_index], color=color, s=24, zorder=3)
    axis.annotate(
        f"{rss_gib[peak_index]:.2f} GiB",
        (times_seconds[peak_index], rss_gib[peak_index]),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )
    axis.set_title(PRECISION_LABELS[precision], fontsize=10)
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Process RSS (GiB)")
    axis.grid(axis="y", alpha=0.2)


def _plot_mode_timelines(mode: str) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, context_length in zip(axes, CONTEXT_LENGTHS, strict=True):
        steps = 3 if mode == "full" else 1
        _plot_timeline(axis, _load_steps(context_length, mode, "fp32", steps), "fp32")
        axis.set_title(f"S={context_length}")
        axis.text(
            0.01,
            0.02,
            f"large, B=1, FP32, steps={steps}",
            transform=axis.transAxes,
            fontsize=8,
            color="#334155",
        )
    title_mode = "inference-only forward" if mode == "forward" else "full training step"
    step_note = "" if mode == "forward" else " (3 steps)"
    figure.suptitle(f"Large model CPU RSS timeline: {title_mode}{step_note}", fontsize=15)
    output_path = OUTPUT_DIR / f"large_{mode}_rss_timeline.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)
    _clean_svg(output_path)
    return output_path


def _plot_peak_summary() -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x_positions = np.arange(len(CONTEXT_LENGTHS))
    for axis, mode in zip(axes, ("forward", "full"), strict=True):
        peaks = [_load(context_length, mode, "fp32")["summary"]["peak_rss_bytes"] / 1024**3 for context_length in CONTEXT_LENGTHS]
        bars = axis.bar(x_positions, peaks, width=0.55, color=PRECISION_COLORS["fp32"])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in peaks], padding=3, fontsize=8)
        axis.set_title("Inference forward" if mode == "forward" else "Full training step")
        axis.set_xticks(x_positions, [f"S={context_length}" for context_length in CONTEXT_LENGTHS])
        axis.set_ylabel("Peak process RSS (GiB)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Large model peak CPU memory (B=1, FP32)", fontsize=15)
    output_path = OUTPUT_DIR / "large_peak_rss_comparison.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)
    _clean_svg(output_path)
    return output_path


def _plot_block_saved_tensors() -> Path:
    result = json.loads((INPUT_DIR / "large_block_s2048_fp32.json").read_text())
    top_five = result["source_summary"][:5]
    labels = [BLOCK_SOURCE_LABELS.get(item["source"], item["source"]) for item in reversed(top_five)]
    values_mib = [item["logical_saved_bytes"] / 1024**2 for item in reversed(top_five)]
    percentages = [item["logical_saved_percent"] for item in reversed(top_five)]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    bars = axes[0].barh(labels, values_mib, color=("#64748b", "#0f766e", "#2563eb", "#7e22ce", "#c2410c"))
    axes[0].bar_label(
        bars,
        labels=[f"{value:.1f} MiB ({percent:.1f}%)" for value, percent in zip(values_mib, percentages, strict=True)],
        padding=4,
        fontsize=8,
    )
    axes[0].set_xlabel("Logical saved-tensor references (MiB)")
    axes[0].set_title("Five largest source operations")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].set_xlim(0, max(values_mib) * 1.35)

    memory_labels = ("Logical references", "Unique storage", "Parameter gradients")
    memory_values = (
        result["logical_non_parameter_saved_tensor_bytes"] / 1024**2,
        result["unique_non_parameter_storage_bytes"] / 1024**2,
        result["parameter_gradient_bytes"] / 1024**2,
    )
    memory_bars = axes[1].bar(memory_labels, memory_values, color=("#7e22ce", "#2563eb", "#dc2626"))
    axes[1].bar_label(memory_bars, labels=[f"{value:.1f} MiB" for value in memory_values], padding=4)
    axes[1].set_ylabel("Memory (MiB)")
    axes[1].set_title("One large TransformerBlock")
    axes[1].tick_params(axis="x", labelrotation=12)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, max(memory_values) * 1.18)

    figure.suptitle("Saved tensors in one block: B=1, S=2048, FP32", fontsize=15)
    output_path = OUTPUT_DIR / "large_block_saved_tensors.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)
    _clean_svg(output_path)
    return output_path


def _plot_three_step_timeline() -> Path:
    result = _load_steps(2048, "full", "fp32", 3)
    samples = result["samples"]
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    _plot_timeline(axis, result, "fp32")
    axis.set_title("Large model, B=1, S=2048, FP32: three full training steps")
    for step in (1, 2, 3):
        matching_times = [sample["elapsed_ms"] / 1_000 for sample in samples if sample["step"] == step]
        left = min(matching_times)
        right = max(matching_times)
        axis.axvline(left, color="#64748b", linestyle="--", linewidth=0.8)
        axis.text(
            (left + right) / 2,
            0.92,
            f"step {step}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    figure.legend(
        handles=[Patch(facecolor=STAGE_COLORS[stage], alpha=0.55, label=stage) for stage in ("zero_grad", "forward", "backward", "optimizer")],
        ncol=4,
        loc="lower center",
        frameon=False,
    )
    output_path = OUTPUT_DIR / "large_s2048_three_step_rss_timeline.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)
    _clean_svg(output_path)
    return output_path


def main() -> None:
    for mode in ("forward", "full"):
        print(f"output_svg={_plot_mode_timelines(mode)}")
    print(f"output_svg={_plot_peak_summary()}")
    print(f"output_svg={_plot_block_saved_tensors()}")
    print(f"output_svg={_plot_three_step_timeline()}")


if __name__ == "__main__":
    main()
