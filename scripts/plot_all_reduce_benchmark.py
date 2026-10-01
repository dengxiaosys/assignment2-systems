"""Summarize and plot a completed all-reduce suite without rerunning collectives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from cs336_systems.all_reduce_benchmark import AllReduceStatistics


def aggregate_suite(suite_dir: Path) -> dict:
    manifest = json.loads((suite_dir / "summary.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("suite must finish successfully before plotting")
    groups = defaultdict(list)
    for entry in manifest["cases"]:
        path = suite_dir / entry["file"]
        case = json.loads(path.read_text())
        if case["status"] != "passed":
            raise ValueError(f"case is not valid: {path}")
        case["source_file"] = path.name
        case["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[(entry["payload_bytes"], entry["world_size"])].append(case)

    cells = []
    for (payload_bytes, world_size), cases in sorted(groups.items()):
        cases.sort(key=lambda case: case["repeat"])
        if [case["repeat"] for case in cases] != list(range(manifest["repeats"])):
            raise ValueError("missing or duplicate repeats")
        samples = [sample for case in cases for sample in case["rank_max_samples_ms"]]
        stats = AllReduceStatistics.summarize_samples(samples)
        bandwidths = AllReduceStatistics.bandwidths(
            payload_bytes=payload_bytes,
            mean_ms=float(stats["mean_ms"]),
            world_size=world_size,
        )
        cells.append(
            {
                "payload_bytes": payload_bytes,
                "world_size": world_size,
                **stats,
                **bandwidths,
                "repeat_means_ms": [case["rank_max_summary"]["mean_ms"] for case in cases],
                "rank_max_samples_ms": samples,
                "rank_means_ms_by_repeat": [[rank["summary"]["mean_ms"] for rank in case["ranks"]] for case in cases],
                "worker_max_rss_kib": max(rank["max_rss_kib"] for case in cases for rank in case["ranks"]),
                "max_subprocess_wall_seconds": max(case["subprocess_wall_seconds"] for case in cases),
                "sources": [{"file": case["source_file"], "sha256": case["source_sha256"]} for case in cases],
            }
        )
    return {"backend": manifest["backend"], "repeats": manifest["repeats"], "cells": cells, "suite_wall_seconds": manifest["suite_wall_seconds"]}


def write_tables(result: dict, output_dir: Path) -> None:
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ("payload_bytes", "world_size", "count", "mean_ms", "std_ms", "median_ms", "p95_ms", "cv_percent", "algbw_GBps", "normalized_busbw_GBps")
    with (output_dir / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["cells"])
    rows = [
        "| Payload | Processes | Mean ± std (ms) | Median (ms) | p95 (ms) | CV (%) | algbw (GB/s) | normalized busbw (GB/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        rows.append(
            f"| {cell['payload_bytes'] // 1_000_000} MB | {cell['world_size']} | "
            f"{cell['mean_ms']:.3f} ± {cell['std_ms']:.3f} | {cell['median_ms']:.3f} | "
            f"{cell['p95_ms']:.3f} | {cell['cv_percent']:.1f} | {cell['algbw_GBps']:.3f} | {cell['normalized_busbw_GBps']:.3f} |"
        )
    (output_dir / "results_table.md").write_text("\n".join(rows) + "\n")


def plot(result: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    process_counts = sorted({cell["world_size"] for cell in result["cells"]})
    for group_index, world_size in enumerate(process_counts):
        cells = [cell for cell in result["cells"] if cell["world_size"] == world_size]
        sizes = [cell["payload_bytes"] / 1e6 for cell in cells]
        color = colors[group_index % len(colors)]
        label = f"{world_size} processes"
        for cell, size in zip(cells, sizes, strict=True):
            samples = cell["rank_max_samples_ms"]
            # Small deterministic horizontal spread makes repeated samples visible.
            xs = [size * 10 ** (0.025 * ((index % 11) - 5) / 5) for index in range(len(samples))]
            axes[0].scatter(xs, samples, color=color, alpha=0.16, s=10, rasterized=False)
        axes[0].plot(sizes, [cell["mean_ms"] for cell in cells], "-o", color=color, label=label)
        axes[1].plot(sizes, [cell["normalized_busbw_GBps"] for cell in cells], "-o", color=color, label=label)
        for cell, size in zip(cells, sizes, strict=True):
            per_repeat_bw = [
                AllReduceStatistics.bandwidths(
                    payload_bytes=cell["payload_bytes"],
                    mean_ms=mean,
                    world_size=world_size,
                )["normalized_busbw_GBps"]
                for mean in cell["repeat_means_ms"]
            ]
            axes[1].scatter([size] * len(per_repeat_bw), per_repeat_bw, color=color, marker="x", s=40)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max latency across ranks per iteration (ms)")
    axes[0].set_title("Latency: raw samples and pooled mean")
    axes[1].set_ylabel("Normalized bus bandwidth (GB/s)")
    axes[1].set_title("Bandwidth: pooled mean; crosses = fresh runs")
    sizes = sorted({cell["payload_bytes"] / 1e6 for cell in result["cells"]})
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(sizes, [f"{size:g}" for size in sizes])
        axis.set_xlabel("Payload per process (decimal MB)")
        axis.grid(alpha=0.25, which="major")
        axis.legend()
    fig.suptitle(f"Single-node {result['backend'].upper()} FP32 SUM all-reduce · {result['repeats']} fresh runs per configuration")
    fig.savefig(output_dir / "all_reduce_scaling.svg", metadata={"Date": None})
    fig.savefig(output_dir / "all_reduce_scaling.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("notes/assets/all_reduce_benchmark"))
    args = parser.parse_args()
    result = aggregate_suite(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(result, args.output_dir)
    plot(result, args.output_dir)
    print(f"plot_status=passed cells={len(result['cells'])} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
