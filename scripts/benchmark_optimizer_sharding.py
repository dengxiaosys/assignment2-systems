"""Benchmark optimizer-state sharding memory and runtime."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig
from cs336_systems.ddp import DDPVariant
from cs336_systems.optimizer_sharding_accounting import (
    OptimizerShardingAccountingConfig,
    OptimizerVariant,
    build_xl_reference_accounting,
    run_case,
)

OPTIMIZER_VARIANTS = ("baseline", "sharded")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-variants", choices=OPTIMIZER_VARIANTS, nargs="+", default=list(OPTIMIZER_VARIANTS))
    parser.add_argument("--ddp-variant", choices=("naive", "flat", "overlap"), default="flat")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--collective-timeout-seconds", type=int, default=300)
    parser.add_argument("--xl-reference-vocab-size", type=int, default=10_000)
    parser.add_argument("--xl-reference-context-length", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/optimizer_sharding_accounting"),
    )
    return parser


def _model_config(args: argparse.Namespace) -> ModelConfig:
    preset = MODEL_CONFIGS[args.model_size]
    return ModelConfig(
        d_model=args.d_model if args.d_model is not None else preset.d_model,
        d_ff=args.d_ff if args.d_ff is not None else preset.d_ff,
        num_layers=args.num_layers if args.num_layers is not None else preset.num_layers,
        num_heads=args.num_heads if args.num_heads is not None else preset.num_heads,
    )


def _config(args: argparse.Namespace, variant: str) -> OptimizerShardingAccountingConfig:
    return OptimizerShardingAccountingConfig(
        optimizer_variant=cast(OptimizerVariant, variant),
        ddp_variant=cast(DDPVariant, args.ddp_variant),
        backend=args.backend,
        world_size=args.world_size,
        model_size=args.model_size,
        model_config=_model_config(args),
        vocab_size=args.vocab_size,
        global_batch_size=args.global_batch_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        num_threads=args.num_threads,
        seed=args.seed,
        collective_timeout_seconds=args.collective_timeout_seconds,
    )


def _snapshot_by_phase(rank_result: dict[str, Any], phase: str) -> dict[str, Any]:
    matches = [snapshot for snapshot in rank_result["memory_snapshots"] if snapshot["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one {phase} snapshot per rank")
    return matches[0]


def _aggregate_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    phases = (
        "after_model_initialization",
        "before_optimizer_step",
        "after_optimizer_step",
    )
    for variant in OPTIMIZER_VARIANTS:
        matching = [result for result in case_results if result["config"]["optimizer_variant"] == variant]
        if not matching:
            continue

        step_means = [result["rank_max_timing_summaries"]["step_total_ms"]["mean_ms"] for result in matching]
        optimizer_means = [result["rank_max_timing_summaries"]["optimizer_step_ms"]["mean_ms"] for result in matching]
        memory = {}
        for phase in phases:
            memory[phase] = {}
            for metric in (
                "rss_bytes",
                "max_rss_bytes",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "cuda_phase_peak_allocated_bytes",
                "cuda_phase_peak_reserved_bytes",
                "parameter_bytes",
                "gradient_bytes",
                "optimizer_state_tensor_bytes_total",
                "owned_parameter_bytes",
            ):
                repeat_rank_max = []
                for result in matching:
                    values = [_snapshot_by_phase(rank_result, phase)[metric] for rank_result in result["ranks"]]
                    if all(value is not None for value in values):
                        repeat_rank_max.append(max(values))
                memory[phase][metric] = statistics.fmean(repeat_rank_max) if repeat_rank_max else None

        aggregate[variant] = {
            "repeat_count": len(matching),
            "step_mean_ms": statistics.fmean(step_means),
            "step_repeat_std_ms": statistics.pstdev(step_means),
            "optimizer_step_mean_ms": statistics.fmean(optimizer_means),
            "optimizer_step_repeat_std_ms": statistics.pstdev(optimizer_means),
            "memory_rank_max_mean_by_phase": memory,
        }

    if all(variant in aggregate for variant in OPTIMIZER_VARIANTS):
        baseline_step = aggregate["baseline"]["step_mean_ms"]
        sharded_step = aggregate["sharded"]["step_mean_ms"]
        aggregate["comparison"] = {
            "sharded_to_baseline_step_ratio": sharded_step / baseline_step,
            "sharded_step_overhead_percent": 100 * (sharded_step - baseline_step) / baseline_step,
        }
    return aggregate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if min(args.xl_reference_vocab_size, args.xl_reference_context_length) <= 0:
        raise ValueError("xl reference vocabulary size and context length must be positive")
    if len(set(args.optimizer_variants)) != len(args.optimizer_variants):
        raise ValueError("optimizer variants must not contain duplicates")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.num_threads)
    for variant in args.optimizer_variants:
        _config(args, variant).validate()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(repeat, variant) for repeat in range(args.repeats) for variant in args.optimizer_variants]
    random.Random(args.seed).shuffle(schedule)
    suite_start = time.perf_counter()
    case_records = []
    case_results = []
    for index, (repeat, variant) in enumerate(schedule):
        case_name = f"repeat{repeat}_{variant}"
        output_path = args.output_dir / f"{case_name}.json"
        print(f"case_start={case_name} progress={index + 1}/{len(schedule)}", flush=True)
        result = run_case(_config(args, variant), output_path)
        case_results.append(result)
        case_records.append(
            {
                "file": output_path.name,
                "repeat": repeat,
                "optimizer_variant": variant,
                "status": result["status"],
                "case_wall_seconds": result["case_wall_seconds"],
            }
        )
        print(
            f"case_end={case_name} status={result['status']} step_mean_ms={result['rank_max_timing_summaries']['step_total_ms']['mean_ms']:.6f}",
            flush=True,
        )

    summary = {
        "status": "passed" if all(record["status"] == "passed" for record in case_records) else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "schedule": [{"repeat": repeat, "optimizer_variant": variant} for repeat, variant in schedule],
        "cases": case_records,
        "aggregate": _aggregate_cases(case_results),
        "xl_reference_accounting": build_xl_reference_accounting(
            vocab_size=args.xl_reference_vocab_size,
            context_length=args.xl_reference_context_length,
            world_size=args.world_size,
        ),
        "suite_wall_seconds": time.perf_counter() - suite_start,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(f"suite_status={summary['status']} output_dir={args.output_dir}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
