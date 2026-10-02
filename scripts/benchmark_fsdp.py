"""Benchmark FSDP all-gather readiness and emit static xl memory accounting."""

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
from typing import Any

import torch

from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig
from cs336_systems.fsdp_accounting import (
    FSDPAccountingConfig,
    build_xl_fsdp_reference,
    run_fsdp_accounting_case,
)

COMPUTE_DTYPES = {
    "float32": None,
    "float16": torch.float16,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-dtypes", choices=COMPUTE_DTYPES, nargs="+", default=["float32", "float16"])
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
        default=Path("benchmark_results/fsdp"),
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


def _config(args: argparse.Namespace, dtype_name: str) -> FSDPAccountingConfig:
    return FSDPAccountingConfig(
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
        compute_dtype=COMPUTE_DTYPES[dtype_name],
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        num_threads=args.num_threads,
        seed=args.seed,
        collective_timeout_seconds=args.collective_timeout_seconds,
    )


def _per_step_communication(
    case: dict[str, Any],
    phase: str,
) -> tuple[list[float], list[float], list[int]]:
    per_rank_wait = []
    per_rank_ready_fraction = []
    per_rank_communicated_bytes = []
    for rank_result in case["ranks"]:
        wait_samples = []
        ready_samples = []
        communicated_bytes_samples = []
        for sample in rank_result["communication_samples"]:
            records = [record for record in sample["all_gather_records"] if record["phase"] == phase]
            wait_samples.append(sum(record["wait_ms"] for record in records))
            ready_samples.append(sum(record["ready_before_wait"] for record in records) / len(records) if records else 1.0)
            communicated_bytes_samples.append(sum(record["communicated_full_bytes"] for record in records))
        per_rank_wait.append(wait_samples)
        per_rank_ready_fraction.append(ready_samples)
        per_rank_communicated_bytes.append(communicated_bytes_samples)
    rank_max_wait = [max(values) for values in zip(*per_rank_wait, strict=True)]
    rank_min_ready_fraction = [min(values) for values in zip(*per_rank_ready_fraction, strict=True)]
    rank_max_communicated_bytes = [max(values) for values in zip(*per_rank_communicated_bytes, strict=True)]
    return rank_max_wait, rank_min_ready_fraction, rank_max_communicated_bytes


def _aggregate_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for dtype_name in COMPUTE_DTYPES:
        matching = [result for result in case_results if result["config"]["compute_dtype"] == (str(COMPUTE_DTYPES[dtype_name]) if COMPUTE_DTYPES[dtype_name] is not None else None)]
        if not matching:
            continue

        step_means = [result["rank_max_timing_summaries"]["step_total_ms"]["mean_ms"] for result in matching]
        forward_means = [result["rank_max_timing_summaries"]["forward_ms"]["mean_ms"] for result in matching]
        forward_wait_means = []
        backward_wait_means = []
        forward_ready_fractions = []
        forward_communicated_bytes = []
        for result in matching:
            forward_wait, forward_ready, forward_bytes = _per_step_communication(
                result,
                "forward",
            )
            backward_wait, _, _ = _per_step_communication(result, "backward")
            forward_wait_means.append(statistics.fmean(forward_wait))
            backward_wait_means.append(statistics.fmean(backward_wait))
            forward_ready_fractions.append(statistics.fmean(forward_ready))
            forward_communicated_bytes.append(statistics.fmean(forward_bytes))

        aggregate[dtype_name] = {
            "repeat_count": len(matching),
            "step_mean_ms": statistics.fmean(step_means),
            "step_repeat_std_ms": statistics.pstdev(step_means),
            "forward_mean_ms": statistics.fmean(forward_means),
            "forward_repeat_std_ms": statistics.pstdev(forward_means),
            "forward_exposed_all_gather_wait_mean_ms": statistics.fmean(forward_wait_means),
            "forward_exposed_all_gather_wait_repeat_std_ms": statistics.pstdev(forward_wait_means),
            "backward_exposed_all_gather_wait_mean_ms": statistics.fmean(backward_wait_means),
            "forward_ready_before_use_fraction": statistics.fmean(forward_ready_fractions),
            "forward_communicated_full_bytes_per_step": statistics.fmean(forward_communicated_bytes),
        }
    return aggregate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if len(set(args.compute_dtypes)) != len(args.compute_dtypes):
        raise ValueError("compute dtypes must not contain duplicates")
    if min(args.xl_reference_vocab_size, args.xl_reference_context_length) <= 0:
        raise ValueError("xl reference vocabulary size and context length must be positive")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.num_threads)
    for dtype_name in args.compute_dtypes:
        _config(args, dtype_name).validate()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(repeat, dtype_name) for repeat in range(args.repeats) for dtype_name in args.compute_dtypes]
    random.Random(args.seed).shuffle(schedule)
    suite_started = time.perf_counter()
    case_records = []
    case_results = []
    for index, (repeat, dtype_name) in enumerate(schedule):
        case_name = f"repeat{repeat}_{dtype_name}"
        output_path = args.output_dir / f"{case_name}.json"
        print(f"case_start={case_name} progress={index + 1}/{len(schedule)}", flush=True)
        result = run_fsdp_accounting_case(_config(args, dtype_name), output_path)
        case_results.append(result)
        case_records.append(
            {
                "file": output_path.name,
                "repeat": repeat,
                "compute_dtype": dtype_name,
                "status": result["status"],
                "case_wall_seconds": result["case_wall_seconds"],
            }
        )
        print(
            f"case_end={case_name} status={result['status']} step_mean_ms={result['rank_max_timing_summaries']['step_total_ms']['mean_ms']:.6f}",
            flush=True,
        )

    summary = {
        "status": ("passed" if all(record["status"] == "passed" for record in case_records) else "failed"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "schedule": [{"repeat": repeat, "compute_dtype": dtype_name} for repeat, dtype_name in schedule],
        "cases": case_records,
        "aggregate": _aggregate_cases(case_results),
        "xl_reference_accounting": build_xl_fsdp_reference(
            vocab_size=args.xl_reference_vocab_size,
            context_length=args.xl_reference_context_length,
            world_size=args.world_size,
        ),
        "suite_wall_seconds": time.perf_counter() - suite_started,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(f"suite_status={summary['status']} output_dir={args.output_dir}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
