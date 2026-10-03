"""Benchmark GPipe fill-drain training across microbatch counts."""

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

from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig
from cs336_systems.pipeline_parallel_accounting import PipelineAccountingConfig, build_xl_pipeline_reference, run_pipeline_accounting_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4, 8])
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
    parser.add_argument("--xl-reference-global-batch-size", type=int, default=8)
    parser.add_argument("--xl-reference-context-length", type=int, default=512)
    parser.add_argument("--xl-reference-microbatches", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/pipeline_parallel"),
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


def _config(args: argparse.Namespace, num_microbatches: int) -> PipelineAccountingConfig:
    return PipelineAccountingConfig(
        backend=args.backend,
        world_size=args.world_size,
        model_config=_model_config(args),
        vocab_size=args.vocab_size,
        global_batch_size=args.global_batch_size,
        context_length=args.context_length,
        num_microbatches=num_microbatches,
        rope_theta=args.rope_theta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        num_threads=args.num_threads,
        seed=args.seed,
        collective_timeout_seconds=args.collective_timeout_seconds,
    )


def _aggregate_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for num_microbatches in sorted({result["config"]["num_microbatches"] for result in case_results}):
        matching = [result for result in case_results if result["config"]["num_microbatches"] == num_microbatches]
        step_means = [result["rank_max_step_summary"]["mean_ms"] for result in matching]
        first = matching[0]
        token_count = first["config"]["global_batch_size"] * first["config"]["context_length"]
        mean_ms = statistics.fmean(step_means)
        aggregate[str(num_microbatches)] = {
            "repeat_count": len(matching),
            "microbatch_size": first["config"]["global_batch_size"] // num_microbatches,
            "step_mean_ms": mean_ms,
            "step_repeat_std_ms": statistics.pstdev(step_means),
            "tokens_per_second": token_count / (mean_ms / 1e3),
            "ideal_fill_drain_efficiency": first["static_accounting"]["ideal_fill_drain_efficiency"],
            "bidirectional_boundary_bytes_per_batch": first["static_accounting"]["bidirectional_boundary_bytes_per_batch"],
        }
    return aggregate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if len(set(args.microbatches)) != len(args.microbatches):
        raise ValueError("microbatch counts must not contain duplicates")
    if (
        min(
            args.xl_reference_vocab_size,
            args.xl_reference_global_batch_size,
            args.xl_reference_context_length,
            args.xl_reference_microbatches,
        )
        <= 0
    ):
        raise ValueError("xl reference dimensions must be positive")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.num_threads)
    for num_microbatches in args.microbatches:
        _config(args, num_microbatches).validate()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(repeat, num_microbatches) for repeat in range(args.repeats) for num_microbatches in args.microbatches]
    random.Random(args.seed).shuffle(schedule)
    suite_started = time.perf_counter()
    case_records = []
    case_results = []
    for index, (repeat, num_microbatches) in enumerate(schedule):
        case_name = f"repeat{repeat}_microbatches{num_microbatches}"
        output_path = args.output_dir / f"{case_name}.json"
        print(f"case_start={case_name} progress={index + 1}/{len(schedule)}", flush=True)
        result = run_pipeline_accounting_case(_config(args, num_microbatches), output_path)
        case_results.append(result)
        case_records.append(
            {
                "file": output_path.name,
                "repeat": repeat,
                "num_microbatches": num_microbatches,
                "status": result["status"],
                "case_wall_seconds": result["case_wall_seconds"],
            }
        )
        print(
            f"case_end={case_name} status={result['status']} step_mean_ms={result['rank_max_step_summary']['mean_ms']:.6f}",
            flush=True,
        )

    summary = {
        "status": ("passed" if all(record["status"] == "passed" for record in case_records) else "failed"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "schedule": [{"repeat": repeat, "num_microbatches": num_microbatches} for repeat, num_microbatches in schedule],
        "cases": case_records,
        "aggregate": _aggregate_cases(case_results),
        "xl_reference_accounting": build_xl_pipeline_reference(
            world_size=args.world_size,
            vocab_size=args.xl_reference_vocab_size,
            global_batch_size=args.xl_reference_global_batch_size,
            context_length=args.xl_reference_context_length,
            num_microbatches=args.xl_reference_microbatches,
        ),
        "suite_wall_seconds": time.perf_counter() - suite_started,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(f"suite_status={summary['status']} output_dir={args.output_dir}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
