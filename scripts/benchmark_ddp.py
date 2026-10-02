"""Run isolated naive DDP benchmark cases."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

VARIANTS = ("naive",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=("small", "medium", "large", "xl", "10b"), default="xl")
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
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--collective-timeout-seconds", type=int, default=300)
    parser.add_argument("--case-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/ddp"))
    parser.add_argument("--case-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _model_config(args: argparse.Namespace):
    from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig

    preset = MODEL_CONFIGS[args.model_size]
    return ModelConfig(
        d_model=args.d_model if args.d_model is not None else preset.d_model,
        d_ff=args.d_ff if args.d_ff is not None else preset.d_ff,
        num_layers=args.num_layers if args.num_layers is not None else preset.num_layers,
        num_heads=args.num_heads if args.num_heads is not None else preset.num_heads,
    )


def _config(args: argparse.Namespace, variant: str):
    from cs336_systems.ddp_benchmark import DDPBenchmarkConfig
    from cs336_systems.ddp import DDPVariant

    return DDPBenchmarkConfig(
        variant=cast(DDPVariant, variant),
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


def _case_command(args: argparse.Namespace, variant: str, output_path: Path) -> list[str]:
    config = _model_config(args)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--variants",
        variant,
        "--backend",
        args.backend,
        "--world-size",
        str(args.world_size),
        "--model-size",
        args.model_size,
        "--d-model",
        str(config.d_model),
        "--d-ff",
        str(config.d_ff),
        "--num-layers",
        str(config.num_layers),
        "--num-heads",
        str(config.num_heads),
        "--vocab-size",
        str(args.vocab_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--context-length",
        str(args.context_length),
        "--rope-theta",
        str(args.rope_theta),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--num-threads",
        str(args.num_threads),
        "--seed",
        str(args.seed),
        "--collective-timeout-seconds",
        str(args.collective_timeout_seconds),
        "--case-output",
        str(output_path.resolve()),
    ]


def run_suite(args: argparse.Namespace) -> int:
    if args.repeats < 1 or args.case_timeout_seconds < 1:
        raise ValueError("repeats and case timeout must be positive")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must not contain duplicates")
    for variant in args.variants:
        _config(args, variant).validate()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(repeat, variant) for repeat in range(args.repeats) for variant in args.variants]
    random.Random(args.seed).shuffle(schedule)
    summary: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "schedule": [{"repeat": repeat, "variant": variant} for repeat, variant in schedule],
        "cases": [],
        "status": "running",
    }
    suite_start = time.perf_counter()
    for index, (repeat, variant) in enumerate(schedule):
        name = f"repeat{repeat}_{variant}"
        output_path = args.output_dir / f"{name}.json"
        log_path = args.output_dir / f"{name}.log"
        print(f"case_start={name} progress={index + 1}/{len(schedule)}", flush=True)
        started = time.perf_counter()
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                _case_command(args, variant, output_path),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.case_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise

        result = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
        if timed_out or returncode:
            result.update(status="timeout" if timed_out else "error", log_file=log_path.name)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        case = {
            "file": output_path.name,
            "repeat": repeat,
            "variant": variant,
            "status": result["status"],
            "subprocess_wall_seconds": time.perf_counter() - started,
            "returncode": returncode,
        }
        summary["cases"].append(case)
        summary["suite_wall_seconds"] = time.perf_counter() - suite_start
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        latency = result.get("rank_max_summaries", {}).get("step_total_ms", {}).get("mean_ms")
        print(f"case_end={name} status={case['status']} mean_step_ms={latency} wall_seconds={case['subprocess_wall_seconds']:.2f}", flush=True)

    passed = all(case["status"] == "passed" for case in summary["cases"])
    summary["status"] = "passed" if passed else "failed"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"suite_status={summary['status']} output_dir={args.output_dir} elapsed_seconds={summary['suite_wall_seconds']:.2f}", flush=True)
    return 0 if passed else 1


def main() -> int:
    args = build_parser().parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.num_threads)

    if args.case_output is not None:
        if len(args.variants) != 1:
            raise ValueError("single-case mode requires exactly one variant")
        from cs336_systems.ddp_benchmark import run_case

        result = run_case(_config(args, args.variants[0]), args.case_output)
        mean_step_ms = result["rank_max_summaries"]["step_total_ms"]["mean_ms"]
        mean_sync_ms = result["rank_max_summaries"]["gradient_sync_wait_ms"]["mean_ms"]
        print(
            f"case_status={result['status']} variant={args.variants[0]} mean_step_ms={mean_step_ms:.6f} mean_gradient_sync_wait_ms={mean_sync_ms:.6f}",
            flush=True,
        )
        return 0
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
