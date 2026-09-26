from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "benchmark_results" / "cpu_mixed_precision"
OUTPUT_PATH = PROJECT_ROOT / "notes" / "assets" / "mixed_precision_benchmark" / "small_cpu_bfloat16_benchmark.svg"
PRECISIONS = ("fp32", "bfloat16")
COLORS = {"fp32": "#0072B2", "bfloat16": "#D55E00"}


@dataclass(frozen=True)
class RunResult:
    precision: str
    forward_mean_ms: float
    forward_std_ms: float
    backward_mean_ms: float
    backward_std_ms: float
    optimizer_mean_ms: float
    optimizer_std_ms: float
    total_mean_ms: float
    total_std_ms: float
    peak_rss_gib: float


def _as_dict(value: Any, *, source_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source_path}: expected a JSON object")
    return value


def _paths(precision: str) -> tuple[Path, Path]:
    return INPUT_DIR / f"small_full_{precision}.json", INPUT_DIR / f"small_full_{precision}.time"


def _peak_rss_gib(path: Path) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path}: peak RSS not found")
    return int(match.group(1)) / 1024**2


def load_result(precision: str) -> RunResult:
    json_path, time_path = _paths(precision)
    payload = _as_dict(json.loads(json_path.read_text(encoding="utf-8")), source_path=json_path)
    phases = _as_dict(payload["phases"], source_path=json_path)
    forward = _as_dict(phases["forward"], source_path=json_path)
    backward = _as_dict(phases["backward"], source_path=json_path)
    optimizer = _as_dict(phases["optimizer"], source_path=json_path)
    total = _as_dict(phases["total"], source_path=json_path)

    expected_autocast = None if precision == "fp32" else "bfloat16"
    actual = (
        payload["model_size"],
        payload["device"],
        payload["dtype"],
        payload["autocast_dtype"],
        payload["mode"],
        payload["batch_size"],
        payload["context_length"],
        payload["num_cpu_threads"],
        payload["warmup_steps"],
        payload["measurement_steps"],
    )
    expected = ("small", "cpu", "float32", expected_autocast, "full", 1, 64, 28, 3, 5)
    if actual != expected:
        raise ValueError(f"{json_path}: expected configuration {expected}, got {actual}")

    return RunResult(
        precision=precision,
        forward_mean_ms=float(forward["mean_ms"]),
        forward_std_ms=float(forward["std_ms"]),
        backward_mean_ms=float(backward["mean_ms"]),
        backward_std_ms=float(backward["std_ms"]),
        optimizer_mean_ms=float(optimizer["mean_ms"]),
        optimizer_std_ms=float(optimizer["std_ms"]),
        total_mean_ms=float(total["mean_ms"]),
        total_std_ms=float(total["std_ms"]),
        peak_rss_gib=_peak_rss_gib(time_path),
    )


def _annotate_ratios(axis: plt.Axes, x: np.ndarray, numerator: list[float], denominator: list[float], width: float) -> None:
    for position, high, low in zip(x, numerator, denominator, strict=True):
        axis.annotate(
            f"{high / low:.2f}x",
            xy=(position + width / 2, high),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["bfloat16"],
        )


def plot_results(results: list[RunResult]) -> None:
    by_precision = {result.precision: result for result in results}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.36

    phase_names = ("Forward", "Backward", "Optimizer", "Total")
    phase_x = np.arange(len(phase_names), dtype=float)
    fp32 = by_precision["fp32"]
    bf16 = by_precision["bfloat16"]
    fp32_times = [fp32.forward_mean_ms, fp32.backward_mean_ms, fp32.optimizer_mean_ms, fp32.total_mean_ms]
    fp32_errors = [fp32.forward_std_ms, fp32.backward_std_ms, fp32.optimizer_std_ms, fp32.total_std_ms]
    bf16_times = [bf16.forward_mean_ms, bf16.backward_mean_ms, bf16.optimizer_mean_ms, bf16.total_mean_ms]
    bf16_errors = [bf16.forward_std_ms, bf16.backward_std_ms, bf16.optimizer_std_ms, bf16.total_std_ms]
    axes[0].bar(phase_x - width / 2, fp32_times, width, yerr=fp32_errors, capsize=4, color=COLORS["fp32"], label="FP32")
    axes[0].bar(phase_x + width / 2, bf16_times, width, yerr=bf16_errors, capsize=4, color=COLORS["bfloat16"], label="BF16 autocast")
    axes[0].set_xticks(phase_x, phase_names)
    axes[0].set_ylabel("Mean time (ms)")
    axes[0].set_title("Training phase latency")
    axes[0].legend()
    _annotate_ratios(axes[0], phase_x, bf16_times, fp32_times, width)

    memory_values = [fp32.peak_rss_gib, bf16.peak_rss_gib]
    memory_x = np.arange(len(PRECISIONS), dtype=float)
    memory_bars = axes[1].bar(memory_x, memory_values, width=0.55, color=[COLORS[precision] for precision in PRECISIONS])
    axes[1].set_xticks(memory_x, ("FP32", "BF16 autocast"))
    axes[1].set_ylabel("Peak process RSS (GiB)")
    axes[1].set_title("Process memory")
    axes[1].bar_label(memory_bars, labels=[f"{value:.3f} GiB" for value in memory_values], padding=4)

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    figure.suptitle("Small model: CPU BF16 autocast on Xeon Platinum 8336C")
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.83, wspace=0.28)
    figure.text(0.5, 0.04, "Full step: B=1, S=64, FP32 parameters, 28 pinned CPU cores, 3 warmups, 5 measurements", ha="center", fontsize=9)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="svg", metadata={"Date": None})
    plt.close(figure)
    OUTPUT_PATH.write_text("\n".join(line.rstrip() for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    print(f"output_svg={OUTPUT_PATH}")


def main() -> None:
    results = [load_result(precision) for precision in PRECISIONS]
    plot_results(results)


if __name__ == "__main__":
    main()
