from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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


def _cell(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    alpha: float = 0.14,
    fontsize: float = 8,
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.1,
            alpha=alpha,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#202124",
    )


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


def plot_loop_order_lifetime(path: Path) -> None:
    figure, (key_axis, query_axis) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    figure.suptitle(
        "Loop order determines which state can survive for one block lifetime",
        fontsize=14,
        fontweight="bold",
    )
    for axis in (key_axis, query_axis):
        _prepare_axis(axis, xlim=(0, 10), ylim=(0, 7))

    key_axis.set_title(
        "(a) Hypothetical global phases: for b -> parallel a",
        loc="left",
        fontweight="bold",
    )
    key_axis.text(
        5.0,
        6.15,
        "Each key tile requires a new global phase",
        ha="center",
        color=GRAY,
    )
    phase_x = (0.25, 3.55, 6.85)
    for key_tile, x in enumerate(phase_x):
        _box(
            key_axis,
            x,
            5.0,
            2.25,
            0.7,
            f"kernel phase b={key_tile}",
            color=PURPLE,
            fontsize=8,
        )
        _box(
            key_axis,
            x,
            3.55,
            2.25,
            0.85,
            "CTA a=0\nload -> update -> store",
            color=BLUE,
            fontsize=8,
        )
        _box(
            key_axis,
            x,
            2.2,
            2.25,
            0.85,
            "CTA a=1\nload -> update -> store",
            color=BLUE,
            fontsize=8,
        )
        if key_tile < len(phase_x) - 1:
            hbm_x = x + 2.48
            _box(
                key_axis,
                hbm_x,
                2.9,
                0.55,
                1.1,
                "HBM\nstate",
                color=VERMILLION,
                alpha=0.16,
                fontsize=7,
            )
            _arrow(key_axis, (x + 2.25, 3.98), (hbm_x, 3.7), color=VERMILLION)
            _arrow(key_axis, (x + 2.25, 2.62), (hbm_x, 3.15), color=VERMILLION)
            _arrow(
                key_axis,
                (hbm_x + 0.55, 3.7),
                (phase_x[key_tile + 1], 3.98),
                color=VERMILLION,
            )
            _arrow(
                key_axis,
                (hbm_x + 0.55, 3.15),
                (phase_x[key_tile + 1], 2.62),
                color=VERMILLION,
            )
    key_axis.text(
        5.0,
        0.85,
        "Global barrier / kernel boundary between b phases",
        ha="center",
        color=VERMILLION,
        fontweight="bold",
    )
    key_axis.text(
        5.0,
        0.35,
        "m, l, A cannot remain in a terminated CTA",
        ha="center",
        color=GRAY,
    )

    query_axis.set_title(
        "(b) Query owner: parallel a -> for b",
        loc="left",
        fontweight="bold",
    )
    query_axis.text(
        5.0,
        6.15,
        "One CTA survives across every key tile",
        ha="center",
        color=GREEN,
    )
    for query_tile, y in enumerate((4.35, 2.45)):
        query_axis.text(
            0.2,
            y + 0.45,
            f"CTA a={query_tile}",
            ha="left",
            va="center",
            fontweight="bold",
        )
        for key_tile, x in enumerate((1.55, 3.55, 5.55)):
            _box(
                query_axis,
                x,
                y,
                1.45,
                0.9,
                f"load K,V b={key_tile}\nupdate state",
                color=GREEN,
                fontsize=7,
            )
            if key_tile:
                _arrow(query_axis, (x - 0.55, y + 0.45), (x, y + 0.45), color=GREEN)
        _box(
            query_axis,
            7.75,
            y,
            1.7,
            0.9,
            "final store\nO^(a), L^(a)",
            color=BLUE,
            fontsize=8,
        )
        _arrow(query_axis, (7.0, y + 0.45), (7.75, y + 0.45), color=GREEN)
    query_axis.text(
        4.35,
        1.15,
        "Q^(a), m^(a), l^(a), A^(a) stay on chip",
        ha="center",
        color=ORANGE,
        fontweight="bold",
    )
    query_axis.text(
        5.0,
        0.45,
        "Making a a grid index necessarily puts the b loop inside that CTA",
        ha="center",
        color=GRAY,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save(figure, path)


def plot_sequence_parallelism(path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    figure.suptitle(
        "FA2 exposes query tiles as independent thread blocks",
        fontsize=14,
        fontweight="bold",
    )
    for axis in axes:
        _prepare_axis(axis, xlim=(0, 10), ylim=(0, 7))

    def draw_sm_grid(axis: Axes, labels: list[str], *, color: str) -> None:
        for sm_index in range(8):
            row, column = divmod(sm_index, 4)
            x = 0.55 + column * 2.25
            y = 4.35 - row * 2.05
            if sm_index < len(labels):
                _box(
                    axis,
                    x,
                    y,
                    1.75,
                    1.15,
                    f"SM {sm_index}\n{labels[sm_index]}",
                    color=color,
                    fontsize=8,
                )
            else:
                _box(
                    axis,
                    x,
                    y,
                    1.75,
                    1.15,
                    f"SM {sm_index}\nidle",
                    color=GRAY,
                    alpha=0.06,
                    fontsize=8,
                )

    fa1_axis, fa2_axis, owner_axis = axes
    fa1_axis.set_title("(a) FA1: parallel over B x H", loc="left", fontweight="bold")
    draw_sm_grid(fa1_axis, ["head 0", "head 1"], color=PURPLE)
    fa1_axis.text(5.0, 6.05, "B=1, H=2  ->  2 blocks", ha="center")
    fa1_axis.text(
        5.0,
        0.65,
        "Only 2 of 8 toy SMs receive work",
        ha="center",
        color=VERMILLION,
        fontweight="bold",
    )

    fa2_axis.set_title(
        "(b) FA2: parallel over B x H x Tr",
        loc="left",
        fontweight="bold",
    )
    fa2_labels = [f"h{head}, q{query}" for head in range(2) for query in range(4)]
    draw_sm_grid(fa2_axis, fa2_labels, color=GREEN)
    fa2_axis.text(5.0, 6.05, "Tr=4  ->  8 independent blocks", ha="center")
    fa2_axis.text(
        5.0,
        0.65,
        "More schedulable work raises utilization",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )

    owner_axis.set_title(
        "(c) One query-owner block",
        loc="left",
        fontweight="bold",
    )
    _box(owner_axis, 0.35, 5.15, 2.0, 0.9, "fixed Q^(a)", color=BLUE)
    _box(
        owner_axis,
        3.0,
        4.85,
        2.65,
        1.5,
        "on-chip state\nm^(a), l^(a), A^(a)",
        color=ORANGE,
    )
    _box(owner_axis, 0.35, 3.45, 2.0, 0.75, "K^(0), V^(0)", color=PURPLE)
    _box(owner_axis, 0.35, 2.35, 2.0, 0.75, "K^(1), V^(1)", color=PURPLE)
    _box(owner_axis, 0.35, 1.25, 2.0, 0.75, "... K^(b), V^(b)", color=PURPLE)
    _box(
        owner_axis,
        3.0,
        2.1,
        2.65,
        1.45,
        "score tile\nonline-softmax update",
        color=GREEN,
    )
    _box(owner_axis, 7.0, 4.9, 2.3, 1.35, "one final write\nO^(a), L^(a)", color=GREEN)

    for start, end in (
        ((2.35, 5.6), (3.0, 5.6)),
        ((2.35, 3.82), (3.0, 3.0)),
        ((2.35, 2.72), (3.0, 2.82)),
        ((2.35, 1.62), (3.0, 2.55)),
        ((4.3, 3.55), (4.3, 4.85)),
        ((5.65, 5.6), (7.0, 5.6)),
    ):
        _arrow(owner_axis, start, end)
    _arrow(
        owner_axis,
        (5.65, 2.6),
        (2.25, 1.6),
        color=PURPLE,
        connectionstyle="arc3,rad=-0.28",
    )
    owner_axis.text(5.25, 1.0, "stream the next K/V tile", color=PURPLE, ha="center")

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save(figure, path)


def plot_causal_tile_map(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 6.4))
    _prepare_axis(axis, xlim=(0, 10), ylim=(0, 8))
    axis.set_title(
        "Causal attention under query-tile ownership",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    tile_count = 6
    x_start = 2.4
    y_start = 1.5
    tile_size = 0.82
    for query_tile in range(tile_count):
        y = y_start + (tile_count - 1 - query_tile) * tile_size
        axis.text(
            2.05,
            y + tile_size / 2,
            f"CTA a={query_tile}",
            ha="right",
            va="center",
            fontsize=8,
        )
        for key_tile in range(tile_count):
            x = x_start + key_tile * tile_size
            if key_tile < query_tile:
                label, color, alpha = "full", GREEN, 0.18
            elif key_tile == query_tile:
                label, color, alpha = "mask", ORANGE, 0.22
            else:
                label, color, alpha = "skip", GRAY, 0.07
            _cell(
                axis,
                x,
                y,
                tile_size,
                tile_size,
                label,
                color=color,
                alpha=alpha,
                fontsize=7,
            )

    for key_tile in range(tile_count):
        x = x_start + key_tile * tile_size + tile_size / 2
        axis.text(x, 6.85, f"key b={key_tile}", ha="center", fontsize=8, rotation=35)

    _box(axis, 7.9, 5.25, 1.5, 0.65, "full tile", color=GREEN, fontsize=8)
    _box(axis, 7.9, 4.15, 1.5, 0.65, "diagonal mask", color=ORANGE, fontsize=8)
    _box(axis, 7.9, 3.05, 1.5, 0.65, "skip entirely", color=GRAY, alpha=0.07, fontsize=8)
    axis.text(
        4.85,
        0.65,
        "Work per query-owner CTA grows from 1 to T_c tiles; causal rows are imbalanced.",
        ha="center",
        color=VERMILLION,
        fontweight="bold",
    )

    figure.tight_layout()
    _save(figure, path)


def plot_backward_ownership(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    _prepare_axis(axis, xlim=(0, 13), ylim=(0, 8))
    axis.set_title(
        "Backward has two orthogonal reduction directions",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    tile_count = 4
    x_start = 1.8
    y_start = 2.05
    tile_size = 0.92
    column_colors = (BLUE, GREEN, ORANGE, PURPLE)
    for query_tile in range(tile_count):
        y = y_start + (tile_count - 1 - query_tile) * tile_size
        axis.text(
            1.45,
            y + tile_size / 2,
            f"query a={query_tile}",
            ha="right",
            va="center",
            fontsize=8,
        )
        for key_tile in range(tile_count):
            x = x_start + key_tile * tile_size
            _cell(
                axis,
                x,
                y,
                tile_size,
                tile_size,
                f"({query_tile},{key_tile})",
                color=column_colors[key_tile],
                alpha=0.12,
                fontsize=7,
            )

    for key_tile in range(tile_count):
        x = x_start + key_tile * tile_size
        _box(
            axis,
            x,
            0.55,
            tile_size,
            0.85,
            f"CTA b={key_tile}\nowns dK,dV",
            color=column_colors[key_tile],
            fontsize=7,
        )
        _arrow(
            axis,
            (x + tile_size / 2, y_start),
            (x + tile_size / 2, 1.4),
            color=column_colors[key_tile],
        )
        axis.text(
            x + tile_size / 2,
            6.35,
            f"key b={key_tile}",
            ha="center",
            fontsize=8,
            rotation=35,
        )

    for query_tile in range(tile_count):
        y = y_start + (tile_count - 1 - query_tile) * tile_size
        _box(
            axis,
            6.35,
            y,
            1.35,
            tile_size,
            f"dQ^(a={query_tile})\natomic sum",
            color=VERMILLION,
            alpha=0.14,
            fontsize=7,
        )
        _arrow(
            axis,
            (x_start + tile_count * tile_size, y + tile_size / 2),
            (6.35, y + tile_size / 2),
            color=VERMILLION,
        )

    _box(
        axis,
        8.35,
        4.55,
        3.8,
        1.3,
        "Column owner scans all query tiles\nand finishes one dK^(b), dV^(b)",
        color=GREEN,
        fontsize=9,
    )
    _box(
        axis,
        8.35,
        2.35,
        3.8,
        1.3,
        "The same CTA only has a partial dQ^(a,b)\nso multiple columns must atomic-add",
        color=VERMILLION,
        alpha=0.14,
        fontsize=9,
    )
    axis.text(
        6.5,
        0.15,
        "Column ownership makes dK/dV race-free, but dQ is a row-wise reduction.",
        ha="center",
        color=GRAY,
        fontweight="bold",
    )

    figure.tight_layout()
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


def plot_sliced_k_reduction(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.2, 6.0))
    _prepare_axis(axis, xlim=(0, 15), ylim=(0, 8))
    axis.set_title(
        "Why sliced-K needs a cross-warp reduction",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    axis.text(
        0.4,
        7.25,
        "One query row, scalar values for illustration",
        color=GRAY,
        fontsize=9,
    )
    _box(
        axis,
        0.4,
        4.0,
        1.55,
        0.85,
        "same query row i",
        color=BLUE,
    )
    _box(
        axis,
        0.35,
        5.75,
        2.0,
        1.0,
        "keys 1-2\np=[0.1, 0.2]\nv=[10, 20]",
        color=PURPLE,
        fontsize=8,
    )
    _box(
        axis,
        0.35,
        1.2,
        2.0,
        1.0,
        "keys 3-4\np=[0.3, 0.4]\nv=[30, 40]",
        color=PURPLE,
        fontsize=8,
    )

    _box(
        axis,
        3.0,
        5.55,
        2.5,
        1.25,
        "warp 0\nu0 = 0.1*10 + 0.2*20\n= 5",
        color=BLUE,
        fontsize=8,
    )
    _box(
        axis,
        3.0,
        1.15,
        2.5,
        1.25,
        "warp 1\nu1 = 0.3*30 + 0.4*40\n= 25",
        color=BLUE,
        fontsize=8,
    )

    _box(
        axis,
        6.2,
        5.65,
        1.9,
        1.0,
        "SMEM slot 0\npartial u0 = 5",
        color=ORANGE,
        fontsize=8,
    )
    _box(
        axis,
        6.2,
        1.25,
        1.9,
        1.0,
        "SMEM slot 1\npartial u1 = 25",
        color=ORANGE,
        fontsize=8,
    )
    _box(
        axis,
        8.8,
        3.45,
        1.75,
        1.05,
        "block barrier\nboth slots ready",
        color=VERMILLION,
        alpha=0.16,
        fontsize=8,
    )
    _box(
        axis,
        11.1,
        3.45,
        1.55,
        1.05,
        "reduce\nu0 + u1",
        color=GRAY,
        fontsize=8,
    )
    _box(
        axis,
        13.2,
        3.45,
        1.4,
        1.05,
        "output\no_i = 30",
        color=GREEN,
        fontsize=8,
    )

    for start, end in (
        ((2.35, 6.25), (3.0, 6.15)),
        ((2.35, 1.7), (3.0, 1.75)),
        ((1.95, 4.55), (3.0, 6.0)),
        ((1.95, 4.3), (3.0, 1.95)),
        ((5.5, 6.15), (6.2, 6.15)),
        ((5.5, 1.75), (6.2, 1.75)),
        ((8.1, 6.15), (8.8, 4.2)),
        ((8.1, 1.75), (8.8, 3.75)),
        ((10.55, 3.98), (11.1, 3.98)),
        ((12.65, 3.98), (13.2, 3.98)),
    ):
        _arrow(axis, start, end)

    axis.text(
        7.5,
        0.35,
        "In the real kernel each partial is a Br x d output tile, not one scalar.",
        ha="center",
        color=VERMILLION,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, path)


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
    plot_loop_order_lifetime(args.output_dir / f"key_major_vs_query_owner_lifetime.{args.format}")
    plot_sequence_parallelism(args.output_dir / f"fa1_fa2_sequence_parallelism.{args.format}")
    plot_causal_tile_map(args.output_dir / f"causal_query_owner_tiles.{args.format}")
    plot_backward_ownership(args.output_dir / f"backward_tile_ownership.{args.format}")
    plot_sliced_k_reduction(args.output_dir / f"sliced_k_partial_reduction.{args.format}")
    plot_work_partition(args.output_dir / f"fa1_vs_fa2_work_partition.{args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
