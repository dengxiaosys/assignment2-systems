"""Single-node FP32 SUM all-reduce with caller-visible timing and raw samples."""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class AllReduceConfig:
    backend: str = "gloo"
    world_size: int = 2
    payload_bytes: int = 1_000_000
    warmup_steps: int = 5
    measurement_steps: int = 20
    num_threads: int = 1
    collective_timeout_seconds: int = 60

    def validate(self) -> None:
        if self.backend not in ("gloo", "nccl"):
            raise ValueError("backend must be gloo or nccl")
        if self.world_size < 2:
            raise ValueError("world_size must be at least 2")
        if self.payload_bytes <= 0 or self.payload_bytes % 4:
            raise ValueError("FP32 payload_bytes must be positive and divisible by 4")
        if self.warmup_steps < 0 or self.measurement_steps < 1:
            raise ValueError("warmup_steps must be nonnegative and measurement_steps positive")
        if self.num_threads < 1 or self.collective_timeout_seconds < 1:
            raise ValueError("num_threads and collective timeout must be positive")
        if not dist.is_available():
            raise RuntimeError("PyTorch distributed is unavailable")
        if self.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("Gloo is unavailable")
        if self.backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < self.world_size):
            raise RuntimeError(f"NCCL requires {self.world_size} distinct visible CUDA devices")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fill_input(tensor: torch.Tensor, pattern: torch.Tensor) -> None:
    """Repeat a small position-dependent pattern without a full-size source copy."""
    full_size = tensor.numel() // pattern.numel() * pattern.numel()
    if full_size:
        tensor[:full_size].view(-1, pattern.numel()).copy_(pattern)
    if full_size < tensor.numel():
        tensor[full_size:].copy_(pattern[: tensor.numel() - full_size])


def _verify_output(tensor: torch.Tensor, expected: torch.Tensor, storage_pointer: int) -> dict[str, Any]:
    """Check every element using bounded temporary memory, outside timing."""
    full_size = tensor.numel() // expected.numel() * expected.numel()
    rows = tensor[:full_size].view(-1, expected.numel())
    for start in range(0, rows.shape[0], 4096):
        if not bool((rows[start : start + 4096] == expected).all().item()):
            raise AssertionError("all-reduce produced an incorrect or non-finite element")
    if full_size < tensor.numel() and not torch.equal(tensor[full_size:], expected[: tensor.numel() - full_size]):
        raise AssertionError("all-reduce tail is incorrect")
    if tensor.data_ptr() != storage_pointer:
        raise AssertionError("all-reduce replaced tensor storage")
    return {"exact_match": True, "checked_elements": tensor.numel(), "in_place": True}


def _worker(rank: int, config: AllReduceConfig, port: int, output_path: str) -> None:
    torch.set_num_threads(config.num_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda", rank) if config.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    timeout = timedelta(seconds=config.collective_timeout_seconds)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=timeout)
    init_kwargs = {"device_id": device} if device.type == "cuda" else {}
    dist.init_process_group(config.backend, store=store, rank=rank, world_size=config.world_size, timeout=timeout, **init_kwargs)
    gathered: list[Any] | None = None
    try:
        tensor = torch.empty(config.payload_bytes // 4, device=device, dtype=torch.float32)
        # Binary fractions and small integers keep the expected FP32 SUM exact.
        base = torch.arange(1024, device=device, dtype=torch.float32).remainder_(17).div_(16)
        pattern = base + rank + 1
        expected = base * config.world_size + config.world_size * (config.world_size + 1) / 2
        pointer = tensor.data_ptr()

        def prepare() -> None:
            _fill_input(tensor, pattern)
            _synchronize(device)
            dist.barrier()
            _synchronize(device)

        # Validate once before warmup, once after warmup, once after measurement.
        prepare()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        _synchronize(device)
        checks = [_verify_output(tensor, expected, pointer)]
        for _ in range(config.warmup_steps):
            prepare()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            _synchronize(device)
        checks.append(_verify_output(tensor, expected, pointer))

        samples_ms = []
        for _ in range(config.measurement_steps):
            prepare()
            start = time.perf_counter_ns()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            _synchronize(device)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        checks.append(_verify_output(tensor, expected, pointer))
        rank_result = _build_rank_result(rank, device, samples_ms, checks)
        # Only small timing/metadata objects are gathered, after all measurements.
        gathered = [None] * config.world_size if rank == 0 else None
        dist.gather_object(rank_result, gathered, dst=0)
    finally:
        dist.destroy_process_group()
    if rank == 0:
        result = {
            "status": "passed",
            "config": asdict(config),
            "ranks": gathered,
            **AllReduceStatistics.summarize_rank_results(gathered, config),
        }
        _write_json(Path(output_path), result)


def run_case(config: AllReduceConfig, output_path: Path) -> dict[str, Any]:
    """Spawn one process per rank, save all raw samples, and return a case record.

    NCCL uses rank r -> cuda:r. Timings include the host collective call and
    device completion wait; reset, barrier, validation, and allocation are excluded.
    The CLI runs this in a subprocess and enforces a whole-case time limit.
    """
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    # Keep the server alive through spawn: binding port=0 avoids a free-port race.
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(_worker, args=(config, server.port, str(output_path.resolve())), nprocs=config.world_size, join=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result.update(_build_case_metadata(config, started_at, time.perf_counter() - wall_start))
    _write_json(output_path, result)
    return result


class AllReduceStatistics:
    """Stateless aggregation for benchmark samples and rank results."""

    @staticmethod
    def summarize_samples(samples_ms: list[float]) -> dict[str, float | int]:
        """Summarize one population; percentiles use linear interpolation."""
        if not samples_ms or any(not math.isfinite(value) or value <= 0 for value in samples_ms):
            raise ValueError("timings must be finite and strictly positive")
        ordered = sorted(samples_ms)
        position = 0.95 * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        mean = statistics.fmean(samples_ms)
        std = statistics.pstdev(samples_ms)
        return {
            "count": len(samples_ms),
            "mean_ms": mean,
            "std_ms": std,
            "median_ms": statistics.median(samples_ms),
            "p95_ms": ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "cv_percent": 100 * std / mean,
        }

    @staticmethod
    def bandwidths(*, payload_bytes: int, mean_ms: float, world_size: int) -> dict[str, float]:
        if payload_bytes <= 0 or mean_ms <= 0 or world_size < 2:
            raise ValueError("bandwidth inputs must be positive and world_size must be at least 2")
        algbw = payload_bytes / (mean_ms / 1000) / 1e9
        return {
            "algbw_GBps": algbw,
            "normalized_busbw_GBps": algbw * 2 * (world_size - 1) / world_size,
        }

    @staticmethod
    def summarize_rank_results(rank_results: list[dict[str, Any]], config: AllReduceConfig) -> dict[str, Any]:
        """Use each iteration's slowest rank as the collective observation."""
        if len(rank_results) != config.world_size:
            raise ValueError("missing rank results")
        by_rank = sorted(rank_results, key=lambda item: item["rank"])
        if [item["rank"] for item in by_rank] != list(range(config.world_size)):
            raise ValueError("duplicate or incorrect ranks")
        if any(len(item["samples_ms"]) != config.measurement_steps for item in by_rank):
            raise ValueError("rank sample counts differ")
        maxima = [max(values) for values in zip(*(item["samples_ms"] for item in by_rank), strict=True)]
        summary = AllReduceStatistics.summarize_samples(maxima)
        return {
            "rank_max_samples_ms": maxima,
            "rank_max_summary": summary,
            **AllReduceStatistics.bandwidths(
                payload_bytes=config.payload_bytes,
                mean_ms=float(summary["mean_ms"]),
                world_size=config.world_size,
            ),
        }


def _build_rank_result(
    rank: int,
    device: torch.device,
    samples_ms: list[float],
    correctness_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "samples_ms": samples_ms,
        "summary": AllReduceStatistics.summarize_samples(samples_ms),
        "correctness_checks": correctness_checks,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_affinity": _cpu_affinity(),
        "torch_num_threads": torch.get_num_threads(),
    }


def _build_case_metadata(
    config: AllReduceConfig,
    started_at_utc: str,
    case_wall_seconds: float,
) -> dict[str, Any]:
    return {
        "started_at_utc": started_at_utc,
        "case_wall_seconds": case_wall_seconds,
        "environment": _collect_environment(config),
        "timing_method": "perf_counter_ns around synchronous all_reduce plus cuda.synchronize for NCCL",
        "sample_population": "per-iteration maximum over ranks; initialization/reset/barrier/checks excluded",
        "payload_unit": "decimal bytes; 1 MB = 1,000,000 bytes; FP32 = 4 bytes",
    }


def _collect_environment(config: AllReduceConfig) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nccl_version": torch.cuda.nccl.version() if config.backend == "nccl" else None,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "gloo_socket_ifname": os.environ.get("GLOO_SOCKET_IFNAME"),
    }


def _cpu_affinity() -> list[int] | None:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
