"""Run isolated all-reduce cases sequentially, retaining failures and raw data."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--payload-mb", type=int, nargs="+", default=[1, 10, 100, 1000], help="Decimal MB, not MiB.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3, help="Fresh process groups for each repetition of each case.")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=240, help="Whole subprocess time limit, including startup.")
    parser.add_argument("--seed", type=int, default=20261001, help="Shuffle case order reproducibly.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/distributed_communication/cpu_gloo"))
    parser.add_argument("--case-output", type=Path, help=argparse.SUPPRESS)
    return parser


def run_suite(args: argparse.Namespace) -> int:
    if args.repeats < 1 or args.timeout_seconds < 1:
        raise ValueError("repeats and timeout_seconds must be positive")
    if len(set(args.world_sizes)) != len(args.world_sizes) or len(set(args.payload_mb)) != len(args.payload_mb):
        raise ValueError("world-sizes and payload-mb must not contain duplicates")
    # Set thread limits before importing torch or starting any worker.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.num_threads)
    from cs336_systems.all_reduce_benchmark import AllReduceConfig

    for world_size in args.world_sizes:
        for payload_mb in args.payload_mb:
            AllReduceConfig(args.backend, world_size, payload_mb * 1_000_000, args.warmup_steps, args.measurement_steps, args.num_threads).validate()
    # No implicit resume or overwriting a previous experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = [(repeat, world_size, payload_mb) for repeat in range(args.repeats) for world_size in args.world_sizes for payload_mb in args.payload_mb]
    random.Random(args.seed).shuffle(schedule)
    summary = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "backend": args.backend,
        "repeats": args.repeats,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "cases": [],
    }
    suite_start = time.perf_counter()
    for index, (repeat, world_size, payload_mb) in enumerate(schedule):
        name = f"repeat{repeat}_p{world_size}_{payload_mb}mb"
        case_path = args.output_dir / f"{name}.json"
        log_path = args.output_dir / f"{name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            args.backend,
            "--world-sizes",
            str(world_size),
            "--payload-mb",
            str(payload_mb),
            "--warmup-steps",
            str(args.warmup_steps),
            "--measurement-steps",
            str(args.measurement_steps),
            "--num-threads",
            str(args.num_threads),
            "--case-output",
            str(case_path.resolve()),
        ]
        print(f"case_start={name} progress={index + 1}/{len(schedule)}", flush=True)
        started = time.perf_counter()
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        result = json.loads(case_path.read_text()) if case_path.exists() else {}
        if timed_out or returncode != 0:
            result.update(status="timeout" if timed_out else "error", log_file=log_path.name)
        result.update(
            repeat=repeat,
            world_size=world_size,
            payload_bytes=payload_mb * 1_000_000,
            subprocess_wall_seconds=time.perf_counter() - started,
            returncode=returncode,
        )
        case_path.write_text(json.dumps(result, indent=2) + "\n")
        summary["cases"].append({"file": case_path.name, "repeat": repeat, "world_size": world_size, "payload_bytes": payload_mb * 1_000_000, "status": result["status"]})
        summary["suite_wall_seconds"] = time.perf_counter() - suite_start
        summary["status"] = "running"
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        latency = result.get("rank_max_summary", {}).get("mean_ms")
        print(f"case_end={name} status={result['status']} mean_rank_max_ms={latency} wall_seconds={result['subprocess_wall_seconds']:.2f}", flush=True)
    passed = all(case["status"] == "passed" for case in summary["cases"])
    summary["status"] = "passed" if passed else "failed"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"suite_status={summary['status']} output_dir={args.output_dir} elapsed_seconds={summary['suite_wall_seconds']:.2f}", flush=True)
    return 0 if passed else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.case_output is not None:
        from cs336_systems.all_reduce_benchmark import AllReduceConfig, run_case

        if len(args.world_sizes) != 1 or len(args.payload_mb) != 1:
            raise ValueError("single-case mode requires one size and one process count")
        config = AllReduceConfig(args.backend, args.world_sizes[0], args.payload_mb[0] * 1_000_000, args.warmup_steps, args.measurement_steps, args.num_threads)
        result = run_case(config, args.case_output)
        print(f"case_status={result['status']} mean_rank_max_ms={result['rank_max_summary']['mean_ms']:.6f}", flush=True)
        return 0
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
