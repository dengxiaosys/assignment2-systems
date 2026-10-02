"""Aggregate DDP benchmark JSON files and render a strategy comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

VARIANT_ORDER = ("naive", "flat", "overlap")
VARIANT_LABELS = {
    "naive": "Naive\nper parameter",
    "flat": "Flat\nsingle buffer",
    "overlap": "Overlap\nper parameter",
}
COLORS = {
    "naive": "#C44E52",
    "flat": "#4C72B0",
    "overlap": "#55A868",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_cases(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = input_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"benchmark suite did not pass: {summary.get('status')}")

    cases = []
    for case_metadata in summary["cases"]:
        if case_metadata["status"] != "passed":
            raise ValueError(f"benchmark case did not pass: {case_metadata}")
        case_path = input_dir / case_metadata["file"]
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["repeat"] = case_metadata["repeat"]
        cases.append(case)

    repeats = {case["repeat"] for case in cases}
    variants = {case["config"]["variant"] for case in cases}
    expected = {(repeat, variant) for repeat in repeats for variant in VARIANT_ORDER}
    actual = {(case["repeat"], case["config"]["variant"]) for case in cases}
    if variants != set(VARIANT_ORDER) or actual != expected or len(cases) != len(expected):
        raise ValueError(f"expected every DDP variant in every repeat, got {sorted(actual)}")
    return summary, cases


def _case_row(case: dict[str, Any]) -> dict[str, Any]:
    step = case["rank_max_summaries"]["step_total_ms"]
    sync = case["rank_max_summaries"]["gradient_sync_wait_ms"]
    return {
        "variant": case["config"]["variant"],
        "repeat": case["repeat"],
        "collective_calls_per_step": case["collective_calls_per_step"],
        "gradient_payload_bytes": case["gradient_payload_bytes"],
        "step_mean_ms": step["mean_ms"],
        "step_std_ms": step["std_ms"],
        "step_median_ms": step["median_ms"],
        "step_p95_ms": step["p95_ms"],
        "sync_wait_mean_ms": sync["mean_ms"],
        "sync_wait_std_ms": sync["std_ms"],
        "sync_wait_fraction_percent": case["gradient_sync_wait_fraction_percent"],
        "sync_timing_semantics": case["gradient_sync_timing_semantics"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in rows if row["variant"] == variant]
        step_means = [float(row["step_mean_ms"]) for row in variant_rows]
        sync_means = [float(row["sync_wait_mean_ms"]) for row in variant_rows]
        fractions = [float(row["sync_wait_fraction_percent"]) for row in variant_rows]
        result[variant] = {
            "step_mean_ms": statistics.fmean(step_means),
            "step_repeat_std_ms": statistics.pstdev(step_means),
            "sync_wait_mean_ms": statistics.fmean(sync_means),
            "sync_wait_repeat_std_ms": statistics.pstdev(sync_means),
            "sync_wait_fraction_percent": statistics.fmean(fractions),
        }

    naive_mean = result["naive"]["step_mean_ms"]
    for values in result.values():
        values["speedup_vs_naive"] = naive_mean / values["step_mean_ms"]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (VARIANT_ORDER.index(row["variant"]), row["repeat"])))


def _write_markdown(path: Path, aggregate: dict[str, dict[str, float]], calls: dict[str, int]) -> None:
    lines = [
        "| Strategy | Calls/step | Step mean +/- repeat std (ms) | Sync/tail mean +/- repeat std (ms) | Sync/tail share | Speedup vs naive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        values = aggregate[variant]
        lines.append(
            f"| `{variant}` | {calls[variant]} | {values['step_mean_ms']:.3f} +/- "
            f"{values['step_repeat_std_ms']:.3f} | {values['sync_wait_mean_ms']:.3f} +/- "
            f"{values['sync_wait_repeat_std_ms']:.3f} | {values['sync_wait_fraction_percent']:.1f}% | "
            f"{values['speedup_vs_naive']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "> `naive` and `flat` report complete post-backward gradient synchronization. `overlap` reports only the exposed post-backward tail wait.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, aggregate: dict[str, dict[str, float]], title: str, subtitle: str) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    x_positions = list(range(len(VARIANT_ORDER)))
    bar_width = 0.34
    step_means = [aggregate[variant]["step_mean_ms"] for variant in VARIANT_ORDER]
    step_errors = [aggregate[variant]["step_repeat_std_ms"] for variant in VARIANT_ORDER]
    sync_means = [aggregate[variant]["sync_wait_mean_ms"] for variant in VARIANT_ORDER]
    sync_errors = [aggregate[variant]["sync_wait_repeat_std_ms"] for variant in VARIANT_ORDER]

    step_bars = axis.bar(
        [position - bar_width / 2 for position in x_positions],
        step_means,
        bar_width,
        yerr=step_errors,
        capsize=4,
        color=[COLORS[variant] for variant in VARIANT_ORDER],
        label="Total step",
    )
    sync_bars = axis.bar(
        [position + bar_width / 2 for position in x_positions],
        sync_means,
        bar_width,
        yerr=sync_errors,
        capsize=4,
        color="#9A9A9A",
        hatch="//",
        label="Post-backward sync / exposed tail",
    )
    axis.bar_label(step_bars, fmt="%.2f", padding=5, fontsize=9)
    axis.bar_label(sync_bars, fmt="%.2f", padding=5, fontsize=9)
    axis.set_xticks(x_positions, [VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
    axis.set_ylabel("Rank-max latency per step (ms)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.01,
        0.98,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    if path.suffix == ".svg":
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    summary, cases = _load_cases(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_case_row(case) for case in cases]
    aggregate = _summarize(rows)
    calls = {row["variant"]: int(row["collective_calls_per_step"]) for row in rows}
    _write_csv(args.output_dir / "results.csv", rows)
    _write_markdown(args.output_dir / "results_table.md", aggregate, calls)
    representative = cases[0]
    repeat_count = len({row["repeat"] for row in rows})
    common_config = {key: value for key, value in representative["config"].items() if key != "variant"}
    backend = representative["config"]["backend"].upper()
    title = f"DDP strategy comparison ({backend})"
    subtitle = (
        f"{representative['config']['world_size']} ranks, {representative['parameter_count']:,} parameters, "
        f"{representative['gradient_payload_bytes']:,}-byte FP32 gradient, "
        f"{repeat_count} independent repeats"
    )
    _plot(args.output_dir / "ddp_strategy_comparison.svg", aggregate, title, subtitle)
    _plot(args.output_dir / "ddp_strategy_comparison.png", aggregate, title, subtitle)

    input_files = [args.input_dir / "summary.json", *(args.input_dir / case["file"] for case in summary["cases"])]
    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_directory": str(args.input_dir),
        "suite_started_at_utc": summary["started_at_utc"],
        "source_command": summary["command"],
        "case_count": len(cases),
        "common_config": common_config,
        "source_sha256": {path.name: _sha256(path) for path in input_files},
        "aggregation": f"mean and population standard deviation across {repeat_count} repeat-level rank-max means",
        "aggregate": aggregate,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_dir={args.output_dir} case_count={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
