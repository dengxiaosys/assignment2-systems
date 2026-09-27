from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)
D_MODELS = (16, 32, 64, 128)


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _set_memory_limit(memory_limit_gib: float) -> int:
    memory_limit_bytes = int(memory_limit_gib * 1024**3)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if hard_limit != resource.RLIM_INFINITY:
        memory_limit_bytes = min(memory_limit_bytes, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    return memory_limit_bytes


def _is_oom_error(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "can't allocate memory",
            "cannot allocate memory",
            "defaultcpuallocator",
            "not enough memory",
            "out of memory",
            "std::bad_alloc",
        )
    )


def _run_worker(args: argparse.Namespace) -> int:
    memory_limit_bytes = _set_memory_limit(args.memory_limit_gib)
    case = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "device": "cpu",
        "seed": args.seed,
    }
    try:
        import torch

        from cs336_systems.attention_benchmark import (
            AttentionBenchmarkConfig,
            benchmark_attention_case,
        )

        torch.set_num_threads(args.num_threads)
        result = benchmark_attention_case(AttentionBenchmarkConfig(**case))
        result["memory_limit_bytes"] = memory_limit_bytes
    except Exception as error:
        result = {
            "status": "oom" if _is_oom_error(error) else "error",
            "config": case,
            "memory_limit_bytes": memory_limit_bytes,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=20),
        }
    _write_json(args.case_output_json, result)
    return 0


def _case_path(output_dir: Path, sequence_length: int, d_model: int) -> Path:
    return output_dir / "cases" / f"s{sequence_length}_d{d_model}.json"


def _worker_command(
    args: argparse.Namespace,
    *,
    sequence_length: int,
    d_model: int,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--batch-size",
        str(args.batch_size),
        "--sequence-length",
        str(sequence_length),
        "--d-model",
        str(d_model),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--memory-limit-gib",
        str(args.memory_limit_gib),
        "--num-threads",
        str(args.num_threads),
        "--seed",
        str(args.seed),
        "--case-output-json",
        str(output_path),
    ]


def _run_case(
    args: argparse.Namespace,
    *,
    sequence_length: int,
    d_model: int,
) -> dict[str, Any]:
    output_path = _case_path(args.output_dir, sequence_length, d_model)
    if output_path.exists() and not args.overwrite:
        return json.loads(output_path.read_text())

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.num_threads)
    environment["MKL_NUM_THREADS"] = str(args.num_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    command = _worker_command(
        args,
        sequence_length=sequence_length,
        d_model=d_model,
        output_path=output_path,
    )
    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        result = {
            "status": "timeout",
            "config": {
                "batch_size": args.batch_size,
                "sequence_length": sequence_length,
                "d_model": d_model,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "device": "cpu",
                "seed": args.seed,
            },
            "memory_limit_bytes": int(args.memory_limit_gib * 1024**3),
            "timeout_seconds": args.timeout_seconds,
            "stdout": "" if error.stdout is None else str(error.stdout)[-4_000:],
            "stderr": "" if error.stderr is None else str(error.stderr)[-4_000:],
        }
        _write_json(output_path, result)
        return result

    if output_path.exists():
        result = json.loads(output_path.read_text())
    else:
        status = "oom" if completed.returncode < 0 else "error"
        result = {
            "status": status,
            "config": {
                "batch_size": args.batch_size,
                "sequence_length": sequence_length,
                "d_model": d_model,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "device": "cpu",
                "seed": args.seed,
            },
            "memory_limit_bytes": int(args.memory_limit_gib * 1024**3),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4_000:],
            "stderr": completed.stderr[-4_000:],
        }
        _write_json(output_path, result)
    return result


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    total_cases = len(SEQUENCE_LENGTHS) * len(D_MODELS)
    case_index = 0
    for sequence_length in SEQUENCE_LENGTHS:
        for d_model in D_MODELS:
            case_index += 1
            print(f"case_start={case_index}/{total_cases} sequence_length={sequence_length} d_model={d_model}")
            result = _run_case(
                args,
                sequence_length=sequence_length,
                d_model=d_model,
            )
            cases.append(result)
            print(f"case_complete={case_index}/{total_cases} sequence_length={sequence_length} d_model={d_model} status={result['status']}")

    combined = {
        "sweep_config": {
            "batch_size": args.batch_size,
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "d_models": list(D_MODELS),
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "memory_limit_gib": args.memory_limit_gib,
            "timeout_seconds": args.timeout_seconds,
            "num_threads": args.num_threads,
            "seed": args.seed,
        },
        "cases": cases,
    }
    _write_json(args.output_json, combined)
    print(f"output_json={args.output_json}")
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated CPU attention benchmarks with an address-space limit.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--memory-limit-gib", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--num-threads", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/cpu_attention"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/cpu_attention/sweep.json"),
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--d-model", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--case-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        min(
            args.batch_size,
            args.measurement_steps,
            args.memory_limit_gib,
            args.timeout_seconds,
            args.num_threads,
        )
        <= 0
    ):
        raise ValueError("benchmark limits and positive dimensions must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup steps cannot be negative")
    if args.worker:
        if args.sequence_length is None or args.d_model is None or args.case_output_json is None:
            raise ValueError("worker mode requires sequence length, d_model, and case output")
        return _run_worker(args)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
