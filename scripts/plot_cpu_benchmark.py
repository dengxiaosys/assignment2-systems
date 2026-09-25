from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "benchmark_results" / "cpu"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "notes" / "assets" / "cpu_benchmark"

MODEL_ORDER = ("small", "medium", "large")
ALL_MODEL_ORDER = ("small", "medium", "large", "xl", "10b")
WARMUP_ORDER = (0, 1, 2, 5)
PHASE_ORDER = ("forward", "loss", "backward", "optimizer")

# Okabe-Ito: color-vision-deficiency-safe and still distinguishable in print.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
MODEL_COLORS = {"small": BLUE, "medium": GREEN, "large": VERMILLION}
PHASE_COLORS = {
    "forward": BLUE,
    "loss": ORANGE,
    "backward": GREEN,
    "optimizer": PURPLE,
}
MODE_COLORS = {
    "full": BLUE,
    "forward-backward": ORANGE,
    "forward": VERMILLION,
}
MODE_MARKERS = {
    "full": "o",
    "forward-backward": "^",
    "forward": "s",
}


@dataclass(frozen=True)
class PhaseStats:
    mean_ms: float
    std_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    source_path: Path
    model_size: str
    mode: str
    device: str
    parameter_count: int
    warmup_steps: int
    measurement_steps: int
    batch_size: int
    context_length: int
    num_cpu_threads: int
    dtype: str
    phases: dict[str, PhaseStats]


def _as_dict(value: Any, *, field: str, source_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source_path}: {field} must be an object")
    return value


def load_result(path: Path) -> BenchmarkResult:
    with path.open(encoding="utf-8") as file:
        payload = _as_dict(json.load(file), field="root", source_path=path)

    phases_payload = _as_dict(payload["phases"], field="phases", source_path=path)
    phases: dict[str, PhaseStats] = {}
    for phase_name, raw_stats in phases_payload.items():
        stats = _as_dict(raw_stats, field=f"phases.{phase_name}", source_path=path)
        samples = tuple(float(value) for value in stats["samples_ms"])
        if len(samples) != int(payload["measurement_steps"]):
            raise ValueError(f"{path}: phase {phase_name!r} has {len(samples)} samples, expected {payload['measurement_steps']}")
        mean_ms = float(stats["mean_ms"])
        std_ms = float(stats["std_ms"])
        if not np.isclose(mean_ms, np.mean(samples)) or not np.isclose(std_ms, np.std(samples)):
            raise ValueError(f"{path}: phase {phase_name!r} summary does not match its samples")
        phases[phase_name] = PhaseStats(
            mean_ms=mean_ms,
            std_ms=std_ms,
            samples_ms=samples,
        )

    return BenchmarkResult(
        source_path=path,
        model_size=str(payload["model_size"]),
        mode=str(payload["mode"]),
        device=str(payload["device"]),
        parameter_count=int(payload["parameter_count"]),
        warmup_steps=int(payload["warmup_steps"]),
        measurement_steps=int(payload["measurement_steps"]),
        batch_size=int(payload["batch_size"]),
        context_length=int(payload["context_length"]),
        num_cpu_threads=int(payload["num_cpu_threads"]),
        dtype=str(payload["dtype"]),
        phases=phases,
    )


def validate_report_configuration(results: list[BenchmarkResult]) -> None:
    expected = ("cpu", "float32", 4, 512, 10, 50)
    for result in results:
        actual = (
            result.device,
            result.dtype,
            result.batch_size,
            result.context_length,
            result.measurement_steps,
            result.num_cpu_threads,
        )
        if actual != expected:
            raise ValueError(f"{result.source_path}: expected (device, dtype, B, S, measurements, threads)={expected}, got {actual}")


def load_full_results(input_dir: Path) -> list[BenchmarkResult]:
    results = [load_result(input_dir / f"{model}_w5.json") for model in MODEL_ORDER]
    for result, expected_model in zip(results, MODEL_ORDER, strict=True):
        if result.model_size != expected_model or result.mode != "full" or result.warmup_steps != 5:
            raise ValueError(f"{result.source_path}: expected {expected_model} full-mode w=5 result")
    return results


def load_warmup_results(input_dir: Path) -> dict[str, list[BenchmarkResult]]:
    results = {model: [load_result(input_dir / f"{model}_w{warmup}.json") for warmup in WARMUP_ORDER] for model in MODEL_ORDER}
    for model, model_results in results.items():
        for result, expected_warmup in zip(model_results, WARMUP_ORDER, strict=True):
            if result.model_size != model or result.mode != "full" or result.warmup_steps != expected_warmup:
                raise ValueError(f"{result.source_path}: unexpected warmup comparison metadata")
    return results


def load_all_w5_results(input_dir: Path, full_results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    extra_results = [
        load_result(input_dir / "xl_forward_backward_w5.json"),
        load_result(input_dir / "10b_forward_w5.json"),
    ]
    results = [*full_results, *extra_results]
    expected_modes = {
        "small": "full",
        "medium": "full",
        "large": "full",
        "xl": "forward-backward",
        "10b": "forward",
    }
    for result, expected_model in zip(results, ALL_MODEL_ORDER, strict=True):
        if result.model_size != expected_model or result.mode != expected_modes[expected_model] or result.warmup_steps != 5:
            raise ValueError(f"{result.source_path}: unexpected w=5 scaling metadata")
    return results


def parse_peak_rss_gib(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{path}: maximum resident set size was not found")
    return int(match.group(1)) / 1024**2


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "cs336-cpu-benchmark",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "xtick.labelsize": 9,
            "ytick.color": "#333333",
            "ytick.labelsize": 9,
        }
    )


def style_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def fixed_offsets(count: int, width: float = 0.22) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def add_box_and_points(
    ax: Axes,
    values: list[float],
    position: float,
    color: str,
    *,
    width: float = 0.5,
) -> None:
    box = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.18, "linewidth": 1.2},
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
    )
    for patch in box["boxes"]:
        patch.set_zorder(1)
    ax.scatter(
        position + fixed_offsets(len(values), width * 0.34),
        values,
        s=24,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.82,
        zorder=3,
    )


def save_svg(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def plot_full_training_profile(results: list[BenchmarkResult], output_path: Path) -> None:
    fig, (latency_ax, share_ax) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        gridspec_kw={"width_ratios": (1.0, 1.25), "wspace": 0.32},
    )

    positions = np.arange(len(results), dtype=float)
    for position, result in zip(positions, results, strict=True):
        total_seconds = [value / 1000 for value in result.phases["total"].samples_ms]
        add_box_and_points(latency_ax, total_seconds, position, MODEL_COLORS[result.model_size])
        latency_ax.scatter(
            position,
            result.phases["total"].mean_ms / 1000,
            marker="D",
            s=34,
            facecolor="white",
            edgecolor="#111111",
            linewidth=1.1,
            zorder=4,
        )

    latency_ax.set_xticks(
        positions,
        labels=[f"{result.model_size.title()}\n{result.parameter_count / 1e6:.1f}M params" for result in results],
    )
    latency_ax.set_ylabel("Total step latency (s)")
    latency_ax.set_title("(a) Measured full-step distribution", loc="left", fontweight="bold")
    latency_ax.set_ylim(bottom=0)
    style_axis(latency_ax)

    y_positions = np.arange(len(results), dtype=float)
    left = np.zeros(len(results))
    for phase in PHASE_ORDER:
        shares = np.array([100 * result.phases[phase].mean_ms / result.phases["total"].mean_ms for result in results])
        bars = share_ax.barh(
            y_positions,
            shares,
            left=left,
            height=0.56,
            label=phase.title(),
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, share in zip(bars, shares, strict=True):
            if share >= 5:
                share_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
        left += shares

    share_ax.set_yticks(
        y_positions,
        labels=[f"{result.model_size.title()} ({result.phases['total'].mean_ms / 1000:.2f} s mean)" for result in results],
    )
    share_ax.invert_yaxis()
    share_ax.set_xlim(0, 100)
    share_ax.set_xlabel("Share of mean total latency (%)")
    share_ax.set_title("(b) Mean step-time composition", loc="left", fontweight="bold")
    legend_handles, legend_labels = share_ax.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center",
        bbox_to_anchor=(0.76, 0.925),
        ncol=4,
        frameon=False,
        handlelength=1.4,
    )
    style_axis(share_ax, grid_axis="x")

    fig.suptitle("CPU full-training benchmark", x=0.06, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.93,
        "FP32 | B=4 | S=512 | warmup=5 | n=10 measurements | lower is better",
        ha="left",
        color=GRAY,
        fontsize=9.5,
    )
    fig.text(
        0.06,
        0.025,
        "Boxes show median and IQR; points are all measurements; white diamonds are means. Phase shares use phase means.",
        ha="left",
        color=GRAY,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.80)
    save_svg(fig, output_path)


def plot_warmup_sensitivity(
    results_by_model: dict[str, list[BenchmarkResult]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.45), sharey=True, gridspec_kw={"wspace": 0.12})
    y_ticks = (0.5, 1.0, 2.0, 4.0, 8.0)

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        model_results = results_by_model[model]
        baseline_samples = model_results[-1].phases["total"].samples_ms
        baseline_median_ms = float(np.median(baseline_samples))
        medians: list[float] = []

        for position, result in enumerate(model_results):
            normalized = [value / baseline_median_ms for value in result.phases["total"].samples_ms]
            add_box_and_points(ax, normalized, float(position), MODEL_COLORS[model], width=0.48)
            medians.append(float(np.median(normalized)))

        ax.plot(
            range(len(model_results)),
            medians,
            color="#111111",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.1,
            zorder=4,
        )
        ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0, zorder=0)
        ax.set_xticks(range(len(model_results)), labels=[str(warmup) for warmup in WARMUP_ORDER])
        ax.set_xlabel("Warmup steps")
        ax.set_title(
            f"{model.title()}\nw=5 median: {baseline_median_ms / 1000:.2f} s",
            fontweight="bold",
        )
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.42, 10)
        ax.set_yticks(y_ticks, labels=[f"{tick:g}x" for tick in y_ticks])
        style_axis(ax)

    axes[0].set_ylabel("Total latency / w=5 median")
    fig.suptitle("Warmup sensitivity of full-training latency", x=0.06, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.925,
        "Each panel is normalized to that model's w=5 median; log2 scale | n=10 per condition",
        ha="left",
        color=GRAY,
        fontsize=9.5,
    )
    fig.text(
        0.06,
        0.025,
        "Points retain every timing sample; boxes show median and IQR; connected markers track group medians. The dashed line is the w=5 baseline.",
        ha="left",
        color=GRAY,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.78)
    save_svg(fig, output_path)


def plot_scaling_and_memory(results: list[BenchmarkResult], output_path: Path) -> None:
    fig, (latency_ax, memory_ax) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        gridspec_kw={"width_ratios": (1.0, 1.05), "wspace": 0.30},
    )

    full_results = [result for result in results if result.mode == "full"]
    parameter_billions = np.array([result.parameter_count / 1e9 for result in full_results])
    means_seconds = np.array([result.phases["total"].mean_ms / 1000 for result in full_results])
    std_seconds = np.array([result.phases["total"].std_ms / 1000 for result in full_results])

    for result, x_value in zip(full_results, parameter_billions, strict=True):
        samples_seconds = np.array(result.phases["total"].samples_ms) / 1000
        x_offsets = np.exp(fixed_offsets(len(samples_seconds), 0.035))
        latency_ax.scatter(
            x_value * x_offsets,
            samples_seconds,
            s=22,
            color=MODEL_COLORS[result.model_size],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.72,
            zorder=2,
        )

    latency_ax.plot(parameter_billions, means_seconds, color=GRAY, linewidth=1.2, zorder=1)
    for result, x_value, mean, std in zip(
        full_results,
        parameter_billions,
        means_seconds,
        std_seconds,
        strict=True,
    ):
        latency_ax.errorbar(
            x_value,
            mean,
            yerr=std,
            fmt="D",
            markersize=6,
            color=MODEL_COLORS[result.model_size],
            markeredgecolor="#111111",
            markeredgewidth=0.7,
            capsize=4,
            linewidth=1.3,
            zorder=4,
        )
        latency_ax.annotate(
            result.model_size.title(),
            (x_value, mean),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    latency_ax.set_xscale("log")
    latency_ax.set_yscale("log", base=2)
    latency_ax.set_xticks(parameter_billions, labels=[f"{value:.2f}" for value in parameter_billions])
    latency_ax.xaxis.set_minor_formatter(NullFormatter())
    latency_ax.set_yticks((1, 2, 4, 8, 16, 32), labels=("1", "2", "4", "8", "16", "32"))
    latency_ax.set_xlabel("Parameters (billions, log scale)")
    latency_ax.set_ylabel("Total step latency (s, log2 scale)")
    latency_ax.set_title("(a) Full-mode scaling", loc="left", fontweight="bold")
    style_axis(latency_ax)

    all_parameter_billions = np.array([result.parameter_count / 1e9 for result in results])
    parameter_curve = np.geomspace(0.1, 15.0, 200)
    parameter_only_gib = parameter_curve * 1e9 * 4 / 1024**3
    memory_ax.plot(
        parameter_curve,
        parameter_only_gib,
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
        label="FP32 parameters only",
        zorder=1,
    )

    for result, x_value in zip(results, all_parameter_billions, strict=True):
        rss_gib = parse_peak_rss_gib(result.source_path.with_suffix(".time"))
        memory_ax.scatter(
            x_value,
            rss_gib,
            marker=MODE_MARKERS[result.mode],
            s=62,
            color=MODE_COLORS[result.mode],
            edgecolor="#111111",
            linewidth=0.7,
            zorder=3,
        )
        label_positions = {
            "small": (7, 4, "left", "bottom"),
            "medium": (7, 4, "left", "bottom"),
            "large": (7, 4, "left", "bottom"),
            "xl": (7, 7, "left", "bottom"),
            "10b": (-7, -7, "right", "top"),
        }
        x_offset, y_offset, horizontal_alignment, vertical_alignment = label_positions[result.model_size]
        memory_ax.annotate(
            f"{result.model_size.upper()}: {rss_gib:.1f} GiB",
            (x_value, rss_gib),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
            fontsize=8.2,
            fontweight="bold",
        )

    memory_ax.set_xscale("log")
    memory_ax.set_yscale("log", base=2)
    memory_ax.set_xticks(all_parameter_billions, labels=[f"{value:.2g}" for value in all_parameter_billions])
    memory_ax.xaxis.set_minor_formatter(NullFormatter())
    memory_ax.set_yticks(
        (0.5, 1, 2, 4, 8, 16, 32, 64),
        labels=("0.5", "1", "2", "4", "8", "16", "32", "64"),
    )
    memory_ax.set_xlabel("Parameters (billions, log scale)")
    memory_ax.set_ylabel("Peak resident memory (GiB, log2 scale)")
    memory_ax.set_title("(b) Peak process memory by available mode", loc="left", fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.2, label="FP32 parameters only"),
        *[
            Line2D(
                [0],
                [0],
                marker=MODE_MARKERS[mode],
                color="none",
                markerfacecolor=MODE_COLORS[mode],
                markeredgecolor="#111111",
                markersize=7,
                label=mode,
            )
            for mode in ("full", "forward-backward", "forward")
        ],
    ]
    memory_ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    style_axis(memory_ax)

    fig.suptitle("CPU scaling and memory footprint", x=0.06, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.93,
        "FP32 | B=4 | S=512 | warmup=5 | points in (a) are individual measurements; diamonds are mean +/- population SD",
        ha="left",
        color=GRAY,
        fontsize=9.2,
    )
    fig.text(
        0.06,
        0.025,
        "Only identical full-mode runs are compared in (a). XL used forward-backward and 10B used forward-only in (b); their timings are not treated as full-training results.",
        ha="left",
        color=GRAY,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.80)
    save_svg(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready SVG charts for the CPU benchmark report.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing benchmark JSON and .time files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated SVG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()

    full_results = load_full_results(args.input_dir)
    warmup_results = load_warmup_results(args.input_dir)
    all_w5_results = load_all_w5_results(args.input_dir, full_results)
    all_results_by_path = {
        result.source_path: result
        for result in [
            *all_w5_results,
            *(result for model_results in warmup_results.values() for result in model_results),
        ]
    }
    validate_report_configuration(list(all_results_by_path.values()))

    outputs = (
        args.output_dir / "cpu_full_training_profile.svg",
        args.output_dir / "cpu_warmup_sensitivity.svg",
        args.output_dir / "cpu_scaling_and_memory.svg",
    )
    plot_full_training_profile(full_results, outputs[0])
    plot_warmup_sensitivity(warmup_results, outputs[1])
    plot_scaling_and_memory(all_w5_results, outputs[2])

    for output in outputs:
        print(f"output_path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
