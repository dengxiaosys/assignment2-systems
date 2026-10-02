"""Single-node benchmarks for the Assignment 2 naive and flat DDP implementations."""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.cuda import nccl

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import ModelConfig, validate_config
from cs336_systems.ddp import DDPVariant, wrap_ddp


@dataclass(frozen=True)
class DDPBenchmarkConfig:
    variant: DDPVariant
    backend: str
    world_size: int
    model_size: str
    model_config: ModelConfig
    vocab_size: int = 10_000
    global_batch_size: int = 4
    context_length: int = 512
    rope_theta: float = 10_000.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 5
    measurement_steps: int = 10
    num_threads: int = 1
    seed: int = 0
    collective_timeout_seconds: int = 300

    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.world_size

    def validate(self) -> None:
        validate_config(self.model_config)
        if self.backend not in ("gloo", "nccl"):
            raise ValueError("backend must be gloo or nccl")
        if self.world_size < 2:
            raise ValueError("world_size must be at least 2")
        if self.global_batch_size <= 0 or self.global_batch_size % self.world_size:
            raise ValueError("global_batch_size must be positive and divisible by world_size")
        if min(self.vocab_size, self.context_length, self.measurement_steps, self.num_threads) <= 0:
            raise ValueError("vocab_size, context_length, measurement_steps, and num_threads must be positive")
        if self.warmup_steps < 0 or self.collective_timeout_seconds <= 0:
            raise ValueError("warmup_steps must be nonnegative and collective timeout must be positive")
        if not dist.is_available():
            raise RuntimeError("PyTorch distributed is unavailable")
        if self.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("Gloo is unavailable")
        if self.backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < self.world_size):
            raise RuntimeError(f"NCCL requires {self.world_size} distinct visible CUDA devices")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvtx_range(device: torch.device, name: str):
    return torch.cuda.nvtx.range(name) if device.type == "cuda" else nullcontext()


def _run_training_step(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    measure: bool,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    if measure:
        _synchronize(device)
        dist.barrier()
        _synchronize(device)
        step_start = time.perf_counter_ns()

    with _nvtx_range(device, "forward"):
        logits = model(input_ids)
    with _nvtx_range(device, "loss"):
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    with _nvtx_range(device, "backward"):
        loss.backward()

    _synchronize(device)
    sync_start = time.perf_counter_ns()
    with _nvtx_range(device, "finish_gradient_synchronization"):
        model.finish_gradient_synchronization()
        _synchronize(device)
    sync_wait_ms = (time.perf_counter_ns() - sync_start) / 1e6

    with _nvtx_range(device, "optimizer"):
        optimizer.step()
    _synchronize(device)
    if not measure:
        return {}
    return {
        "step_total_ms": (time.perf_counter_ns() - step_start) / 1e6,
        "gradient_sync_wait_ms": sync_wait_ms,
    }


def _worker(rank: int, config: DDPBenchmarkConfig, port: int, output_path: str) -> None:
    torch.set_num_threads(config.num_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda", rank) if config.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    timeout = timedelta(seconds=config.collective_timeout_seconds)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=timeout)
    if device.type == "cuda":
        dist.init_process_group(
            config.backend,
            store=store,
            rank=rank,
            world_size=config.world_size,
            timeout=timeout,
            device_id=device,
        )
    else:
        dist.init_process_group(
            config.backend,
            store=store,
            rank=rank,
            world_size=config.world_size,
            timeout=timeout,
        )
    gathered: list[Any] | None = None
    try:
        torch.manual_seed(config.seed + rank)
        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.model_config.d_model,
            num_layers=config.model_config.num_layers,
            num_heads=config.model_config.num_heads,
            d_ff=config.model_config.d_ff,
            rope_theta=config.rope_theta,
            device=device,
            dtype=torch.float32,
        )
        ddp_model = wrap_ddp(model, config.variant)
        optimizer = AdamW(ddp_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        generator = torch.Generator(device=device).manual_seed(config.seed + 10_000 + rank)
        shape = (config.local_batch_size, config.context_length)
        input_ids = torch.randint(config.vocab_size, shape, generator=generator, device=device)
        targets = torch.randint(config.vocab_size, shape, generator=generator, device=device)

        for _ in range(config.warmup_steps):
            _run_training_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
                measure=False,
            )

        samples = {"step_total_ms": [], "gradient_sync_wait_ms": []}
        with _nvtx_range(device, "ddp_benchmark_measurement"):
            for _ in range(config.measurement_steps):
                step = _run_training_step(
                    model=ddp_model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                    device=device,
                    measure=True,
                )
                for name, value in step.items():
                    samples[name].append(value)

        rank_result = _build_rank_result(rank, device, samples)
        gathered = [None] * config.world_size if rank == 0 else None
        dist.gather_object(rank_result, gathered, dst=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        assert gathered is not None
        result = _build_result(config, model, gathered)
        _write_json(Path(output_path), result)


def run_case(config: DDPBenchmarkConfig, output_path: Path) -> dict[str, Any]:
    """Run one isolated multi-process benchmark case and retain raw rank samples."""

    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(
        _worker,
        args=(config, server.port, str(output_path.resolve())),
        nprocs=config.world_size,
        join=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result.update(
        started_at_utc=started_at,
        case_wall_seconds=time.perf_counter() - wall_start,
        environment=_collect_environment(config),
    )
    _write_json(output_path, result)
    return result


def summarize_samples(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms or any(not math.isfinite(value) or value < 0 for value in samples_ms):
        raise ValueError("timings must be finite and nonnegative")
    ordered = sorted(samples_ms)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    mean = statistics.fmean(samples_ms)
    return {
        "count": len(samples_ms),
        "mean_ms": mean,
        "std_ms": statistics.pstdev(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _build_rank_result(
    rank: int,
    device: torch.device,
    samples: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "samples": samples,
        "summaries": {name: summarize_samples(values) for name, values in samples.items()},
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def _build_result(
    config: DDPBenchmarkConfig,
    model: TransformerLM,
    rank_results: list[dict[str, Any]],
) -> dict[str, Any]:
    rank_results.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in rank_results] != list(range(config.world_size)):
        raise ValueError("rank results are missing or duplicated")

    rank_max_samples = {}
    for name in ("step_total_ms", "gradient_sync_wait_ms"):
        per_rank = [item["samples"][name] for item in rank_results]
        if any(len(samples) != config.measurement_steps for samples in per_rank):
            raise ValueError(f"rank sample counts differ for {name}")
        rank_max_samples[name] = [max(values) for values in zip(*per_rank, strict=True)]

    summaries = {name: summarize_samples(values) for name, values in rank_max_samples.items()}
    step_mean = float(summaries["step_total_ms"]["mean_ms"])
    sync_mean = float(summaries["gradient_sync_wait_ms"]["mean_ms"])
    parameters = tuple(model.parameters())
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    return {
        "status": "passed",
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_tensors": len(trainable),
        "gradient_payload_bytes": sum(parameter.numel() * parameter.element_size() for parameter in trainable),
        "collective_calls_per_step": 1 if config.variant == "flat" else len(trainable),
        "rank_max_samples": rank_max_samples,
        "rank_max_summaries": summaries,
        "gradient_sync_wait_fraction_percent": 100 * sync_mean / step_mean,
        "gradient_sync_timing_semantics": "complete post-backward gradient synchronization",
        "ranks": rank_results,
    }


def _collect_environment(config: DDPBenchmarkConfig) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nccl_version": nccl.version() if config.backend == "nccl" else None,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
