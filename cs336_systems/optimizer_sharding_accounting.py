"""Memory and runtime accounting for replicated and sharded optimizer state."""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.cuda import nccl

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig, validate_config
from cs336_systems.ddp import DDPVariant, wrap_ddp
from cs336_systems.sharded_optimizer import ShardedOptimizer

type OptimizerVariant = Literal["baseline", "sharded"]


@dataclass(frozen=True)
class OptimizerShardingAccountingConfig:
    """Configuration for one isolated distributed accounting case."""

    optimizer_variant: OptimizerVariant
    ddp_variant: DDPVariant
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
    warmup_steps: int = 3
    measurement_steps: int = 10
    num_threads: int = 1
    seed: int = 0
    collective_timeout_seconds: int = 300

    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.world_size

    def validate(self) -> None:
        validate_config(self.model_config)
        if self.optimizer_variant not in ("baseline", "sharded"):
            raise ValueError("optimizer_variant must be baseline or sharded")
        if self.ddp_variant not in ("naive", "flat", "overlap"):
            raise ValueError("ddp_variant must be naive, flat, or overlap")
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


def summarize_samples(samples_ms: list[float]) -> dict[str, float | int]:
    """Summarize synchronized nonnegative timing samples."""

    if not samples_ms or any(not math.isfinite(value) or value < 0 for value in samples_ms):
        raise ValueError("timings must be finite and nonnegative")
    ordered = sorted(samples_ms)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return {
        "count": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def build_static_accounting(
    *,
    model_config: ModelConfig,
    vocab_size: int,
    context_length: int,
    world_size: int,
) -> dict[str, Any]:
    """Calculate persistent FP32 training-state bytes without allocating storage."""

    validate_config(model_config)
    if min(vocab_size, context_length, world_size) <= 0:
        raise ValueError("vocab_size, context_length, and world_size must be positive")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=10_000.0,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    parameters = tuple(model.parameters())
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in parameters)
    rank_to_owned_parameter_bytes = [0] * world_size
    rank_to_owned_parameter_tensors = [0] * world_size
    for parameter in parameters:
        owner_rank = min(
            range(world_size),
            key=lambda rank: (rank_to_owned_parameter_bytes[rank], rank),
        )
        rank_to_owned_parameter_bytes[owner_rank] += _tensor_bytes(parameter)
        rank_to_owned_parameter_tensors[owner_rank] += 1

    baseline_moment_bytes = 2 * parameter_bytes
    sharded_ranks = []
    for rank in range(world_size):
        owned_bytes = rank_to_owned_parameter_bytes[rank]
        owned_tensors = rank_to_owned_parameter_tensors[rank]
        sharded_ranks.append(
            {
                "rank": rank,
                "owned_parameter_tensors": owned_tensors,
                "owned_parameter_bytes": owned_bytes,
                "adamw_moment_bytes": 2 * owned_bytes,
                "adamw_step_scalar_bytes": 4 * owned_tensors,
                "persistent_parameter_gradient_moment_bytes": 2 * parameter_bytes + 2 * owned_bytes,
            }
        )

    return {
        "model_config": asdict(model_config),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "world_size": world_size,
        "dtype": "torch.float32",
        "parameter_tensors": len(parameters),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes_per_rank": parameter_bytes,
        "gradient_bytes_per_rank": parameter_bytes,
        "baseline_per_rank": {
            "adamw_moment_bytes": baseline_moment_bytes,
            "adamw_step_scalar_bytes": 4 * len(parameters),
            "persistent_parameter_gradient_moment_bytes": 4 * parameter_bytes,
        },
        "sharded_ranks": sharded_ranks,
    }


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _read_rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _optimizer_state_tensor_bytes_by_device(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                totals[str(value.device)] += _tensor_bytes(value)
    return dict(sorted(totals.items()))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(
    *,
    phase: str,
    model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    _synchronize(device)
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
    state_bytes_by_device = _optimizer_state_tensor_bytes_by_device(optimizer)
    snapshot: dict[str, Any] = {
        "phase": phase,
        "rss_bytes": _read_rss_bytes(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "parameter_bytes": sum(_tensor_bytes(parameter) for parameter in parameters),
        "gradient_bytes": sum(_tensor_bytes(gradient) for gradient in gradients),
        "gradient_tensors": len(gradients),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_tensor_bytes_by_device": state_bytes_by_device,
        "optimizer_state_tensor_bytes_total": sum(state_bytes_by_device.values()),
    }
    if isinstance(optimizer, ShardedOptimizer):
        snapshot["owned_parameter_bytes"] = sum(
            _tensor_bytes(parameter) for parameter, owner_rank in zip(parameters, optimizer.parameter_owners, strict=True) if owner_rank == dist.get_rank()
        )
    else:
        snapshot["owned_parameter_bytes"] = snapshot["parameter_bytes"]

    if device.type == "cuda":
        snapshot.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_phase_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_phase_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    else:
        snapshot.update(
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_phase_peak_allocated_bytes=None,
            cuda_phase_peak_reserved_bytes=None,
        )
    return snapshot


def _make_optimizer(
    variant: OptimizerVariant,
    parameters,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if variant == "baseline":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=False,
        )
    return ShardedOptimizer(
        parameters,
        torch.optim.AdamW,
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )


def _forward_backward(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    model.finish_gradient_synchronization()
    _synchronize(device)


def _run_first_profiled_step(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    _reset_peak_memory(device)
    _forward_backward(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        device=device,
    )
    before_optimizer = _memory_snapshot(
        phase="before_optimizer_step",
        model=model,
        optimizer=optimizer,
        device=device,
    )

    _reset_peak_memory(device)
    optimizer.step()
    _synchronize(device)
    after_optimizer = _memory_snapshot(
        phase="after_optimizer_step",
        model=model,
        optimizer=optimizer,
        device=device,
    )
    return [before_optimizer, after_optimizer]


def _run_timed_step(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    measure: bool,
) -> dict[str, float]:
    if measure:
        dist.barrier()
        _synchronize(device)
        step_start = time.perf_counter_ns()

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()

    sync_start = time.perf_counter_ns()
    model.finish_gradient_synchronization()
    _synchronize(device)
    gradient_sync_ms = (time.perf_counter_ns() - sync_start) / 1e6

    optimizer_start = time.perf_counter_ns()
    optimizer.step()
    _synchronize(device)
    optimizer_step_ms = (time.perf_counter_ns() - optimizer_start) / 1e6
    if not measure:
        return {}
    return {
        "step_total_ms": (time.perf_counter_ns() - step_start) / 1e6,
        "gradient_sync_ms": gradient_sync_ms,
        "optimizer_step_ms": optimizer_step_ms,
    }


def _worker(
    rank: int,
    config: OptimizerShardingAccountingConfig,
    port: int,
    output_path: str,
) -> None:
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

    gathered: list[dict[str, Any] | None] | None = None
    model = None
    try:
        torch.manual_seed(config.seed + rank)
        _reset_peak_memory(device)
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
        ddp_model = wrap_ddp(model, config.ddp_variant)
        optimizer = _make_optimizer(
            config.optimizer_variant,
            ddp_model.parameters(),
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        memory_snapshots = [
            _memory_snapshot(
                phase="after_model_initialization",
                model=ddp_model,
                optimizer=optimizer,
                device=device,
            )
        ]

        generator = torch.Generator(device=device).manual_seed(config.seed + 10_000 + rank)
        shape = (config.local_batch_size, config.context_length)
        input_ids = torch.randint(config.vocab_size, shape, generator=generator, device=device)
        targets = torch.randint(config.vocab_size, shape, generator=generator, device=device)
        memory_snapshots.extend(
            _run_first_profiled_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
            )
        )

        for _ in range(config.warmup_steps):
            _run_timed_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
                measure=False,
            )

        samples = {
            "step_total_ms": [],
            "gradient_sync_ms": [],
            "optimizer_step_ms": [],
        }
        for _ in range(config.measurement_steps):
            timings = _run_timed_step(
                model=ddp_model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
                measure=True,
            )
            for name, value in timings.items():
                samples[name].append(value)

        rank_result = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "memory_snapshots": memory_snapshots,
            "timing_samples_ms": samples,
            "timing_summaries": {name: summarize_samples(values) for name, values in samples.items()},
        }
        gathered = [None] * config.world_size if rank == 0 else None
        dist.gather_object(rank_result, gathered, dst=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        if model is None or gathered is None or any(result is None for result in gathered):
            raise RuntimeError("rank 0 did not receive every accounting result")
        rank_results = sorted(
            (result for result in gathered if result is not None),
            key=lambda result: result["rank"],
        )
        result = _build_result(config, model, rank_results)
        _write_json(Path(output_path), result)


def run_case(
    config: OptimizerShardingAccountingConfig,
    output_path: Path,
) -> dict[str, Any]:
    """Run one isolated distributed optimizer accounting case."""

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


def _build_result(
    config: OptimizerShardingAccountingConfig,
    model: TransformerLM,
    rank_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if [result["rank"] for result in rank_results] != list(range(config.world_size)):
        raise ValueError("rank results are missing or duplicated")

    rank_max_samples = {}
    for name in ("step_total_ms", "gradient_sync_ms", "optimizer_step_ms"):
        per_rank = [result["timing_samples_ms"][name] for result in rank_results]
        if any(len(samples) != config.measurement_steps for samples in per_rank):
            raise ValueError(f"rank sample counts differ for {name}")
        rank_max_samples[name] = [max(values) for values in zip(*per_rank, strict=True)]

    parameters = tuple(model.parameters())
    return {
        "status": "passed",
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_tensors": len(parameters),
        "parameter_bytes_per_rank": sum(_tensor_bytes(parameter) for parameter in parameters),
        "rank_max_timing_samples_ms": rank_max_samples,
        "rank_max_timing_summaries": {name: summarize_samples(values) for name, values in rank_max_samples.items()},
        "ranks": rank_results,
    }


def _collect_environment(config: OptimizerShardingAccountingConfig) -> dict[str, Any]:
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
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def build_xl_reference_accounting(
    *,
    vocab_size: int,
    context_length: int,
    world_size: int,
) -> dict[str, Any]:
    """Build the handout's FP32 xl persistent-state estimate."""

    return build_static_accounting(
        model_config=MODEL_CONFIGS["xl"],
        vocab_size=vocab_size,
        context_length=context_length,
        world_size=world_size,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
