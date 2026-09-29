from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("notes/assets/flash_attention2")
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#5B6470"
LIGHT_GRAY = "#EEF1F4"


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "cs336-flash-attention2-textbook",
        }
    )


def _prepare_axis(axis: Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.axis("off")


def _box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    alpha: float = 0.13,
    fontsize: float = 9,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor=color,
        edgecolor=color,
        linewidth=1.3,
        alpha=alpha,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#202124",
    )


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    connectionstyle: str = "arc3",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    print(f"output_path={path}")


def plot_io_comparison(path: Path) -> None:
    figure, (naive_axis, flash_axis) = plt.subplots(1, 2, figsize=(13, 4.5))
    figure.suptitle(
        "Attention dataflow: materialized matrices vs. tiled online reduction",
        fontsize=14,
        fontweight="bold",
    )

    for axis in (naive_axis, flash_axis):
        _prepare_axis(axis, xlim=(0, 10), ylim=(0, 6))

    naive_axis.set_title("(a) Naive attention", loc="left", fontweight="bold")
    _box(naive_axis, 0.2, 4.2, 1.4, 0.8, "Q, K\nHBM", color=BLUE)
    _box(naive_axis, 2.1, 4.2, 1.4, 0.8, "GEMM\nQ K^T", color=GRAY)
    _box(
        naive_axis,
        4.0,
        4.05,
        1.65,
        1.1,
        "S: N x N\nwrite/read HBM",
        color=VERMILLION,
        alpha=0.18,
    )
    _box(naive_axis, 6.15, 4.2, 1.2, 0.8, "Softmax", color=ORANGE)
    _box(
        naive_axis,
        7.85,
        4.05,
        1.65,
        1.1,
        "P: N x N\nwrite/read HBM",
        color=VERMILLION,
        alpha=0.18,
    )
    _box(naive_axis, 4.0, 1.65, 1.65, 0.8, "GEMM\nP V", color=GRAY)
    _box(naive_axis, 6.2, 1.65, 1.4, 0.8, "O\nHBM", color=GREEN)
    _box(naive_axis, 1.8, 1.65, 1.4, 0.8, "V\nHBM", color=BLUE)
    for start, end in (
        ((1.6, 4.6), (2.1, 4.6)),
        ((3.5, 4.6), (4.0, 4.6)),
        ((5.65, 4.6), (6.15, 4.6)),
        ((7.35, 4.6), (7.85, 4.6)),
        ((8.7, 4.05), (5.3, 2.45)),
        ((3.2, 2.05), (4.0, 2.05)),
        ((5.65, 2.05), (6.2, 2.05)),
    ):
        _arrow(naive_axis, start, end)
    naive_axis.text(
        5.0,
        0.65,
        "Quadratic HBM traffic and O(N^2) saved activations",
        ha="center",
        color=VERMILLION,
        fontweight="bold",
    )

    flash_axis.set_title("(b) FlashAttention-2 forward", loc="left", fontweight="bold")
    _box(flash_axis, 0.3, 4.35, 1.45, 0.8, "Q tile\nBq x d", color=BLUE)
    _box(flash_axis, 0.3, 2.8, 1.45, 0.8, "K, V tile\nBk x d", color=BLUE)
    _box(
        flash_axis,
        2.55,
        2.65,
        3.15,
        2.55,
        "One fused program\n\nS tile = Q K^T\nonline softmax\nweighted-sum update",
        color=GREEN,
        alpha=0.12,
        fontsize=10,
    )
    _box(
        flash_axis,
        6.45,
        3.25,
        1.75,
        1.45,
        "On-chip state\nm, l, O accumulator",
        color=ORANGE,
    )
    _box(flash_axis, 8.7, 3.8, 1.0, 0.8, "O tile\nHBM", color=GREEN)
    _box(flash_axis, 8.7, 2.45, 1.0, 0.8, "L row\nHBM", color=PURPLE)
    for start, end in (
        ((1.75, 4.75), (2.55, 4.55)),
        ((1.75, 3.2), (2.55, 3.35)),
        ((5.7, 3.95), (6.45, 3.95)),
        ((8.2, 4.25), (8.7, 4.2)),
        ((8.2, 3.55), (8.7, 2.85)),
    ):
        _arrow(flash_axis, start, end)
    _arrow(
        flash_axis,
        (4.9, 2.65),
        (1.4, 2.75),
        color=BLUE,
        connectionstyle="arc3,rad=-0.28",
    )
    flash_axis.text(3.0, 1.35, "stream next K/V tile", color=BLUE, ha="center")
    flash_axis.text(
        5.0,
        0.65,
        "Never materialize full S or P in HBM",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save(figure, path)


def _warp_row(
    axis: Axes,
    y: float,
    warp_index: int,
    left_text: str,
    right_text: str,
) -> None:
    _box(axis, 0.4, y, 1.1, 0.55, f"warp {warp_index}", color=GRAY, fontsize=8)
    _box(axis, 1.8, y, 2.2, 0.55, left_text, color=PURPLE, fontsize=8)
    _box(axis, 6.4, y, 2.2, 0.55, right_text, color=BLUE, fontsize=8)


def plot_work_partition(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.2, 5.6))
    _prepare_axis(axis, xlim=(0, 10), ylim=(0, 7))
    axis.set_title(
        "FlashAttention-2 work partitioning inside one thread block",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(2.9, 6.25, "FlashAttention-1: sliced-K", ha="center", fontweight="bold")
    axis.text(7.5, 6.25, "FlashAttention-2: sliced-Q", ha="center", fontweight="bold")

    for warp_index, y in enumerate((5.25, 4.45, 3.65, 2.85)):
        _warp_row(
            axis,
            y,
            warp_index,
            f"same Q,\nK/V slice {warp_index}",
            f"Q slice {warp_index},\nshared K/V",
        )

    _box(
        axis,
        1.55,
        1.25,
        2.7,
        0.9,
        "Partial outputs must be\nreduced across warps",
        color=VERMILLION,
        alpha=0.16,
    )
    _box(
        axis,
        6.15,
        1.25,
        2.7,
        0.9,
        "Each warp owns independent\noutput rows",
        color=GREEN,
        alpha=0.16,
    )
    for y in (5.52, 4.72, 3.92, 3.12):
        _arrow(axis, (4.0, y), (3.0, 2.15), color=VERMILLION)
        _arrow(axis, (8.6, y), (7.5, 2.15), color=GREEN)

    axis.text(
        2.9,
        0.55,
        "shared-memory exchange + synchronization",
        ha="center",
        color=VERMILLION,
    )
    axis.text(
        7.5,
        0.55,
        "less communication between warps",
        ha="center",
        color=GREEN,
    )
    figure.tight_layout()
    _save(figure, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render FlashAttention-2 textbook diagrams.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_style()
    plot_io_comparison(args.output_dir / f"naive_vs_flash_io.{args.format}")
    plot_work_partition(args.output_dir / f"fa1_vs_fa2_work_partition.{args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
