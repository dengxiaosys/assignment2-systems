"""Static and runtime accounting for pipeline-parallel training."""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.cuda import nccl

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig, validate_config
from cs336_systems.optimizer_sharding_accounting import summarize_samples
from cs336_systems.pipeline_parallel import PipelineParallel, TransformerPipelineStage


@dataclass(frozen=True)
class PipelineAccountingConfig:
    """Configuration for one isolated pipeline-parallel benchmark."""

    backend: str
    world_size: int
    model_config: ModelConfig
    vocab_size: int
    global_batch_size: int
    context_length: int
    num_microbatches: int
    rope_theta: float = 10_000.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 3
    measurement_steps: int = 10
    num_threads: int = 1
    seed: int = 0
    collective_timeout_seconds: int = 300

    @property
    def microbatch_size(self) -> int:
        return self.global_batch_size // self.num_microbatches

    def validate(self) -> None:
        validate_config(self.model_config)
        if self.backend not in ("gloo", "nccl"):
            raise ValueError("backend must be gloo or nccl")
        if self.world_size < 2 or self.world_size > self.model_config.num_layers:
            raise ValueError("world_size must be between 2 and num_layers")
        if min(self.vocab_size, self.context_length, self.num_microbatches, self.measurement_steps, self.num_threads) <= 0:
            raise ValueError("vocab_size, context_length, num_microbatches, measurement_steps, and num_threads must be positive")
        if self.global_batch_size <= 0 or self.global_batch_size % self.num_microbatches:
            raise ValueError("global_batch_size must be positive and divisible by num_microbatches")
        if self.warmup_steps < 0 or self.collective_timeout_seconds <= 0:
            raise ValueError("warmup_steps must be nonnegative and collective timeout must be positive")
        if self.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("Gloo is unavailable")
        if self.backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < self.world_size):
            raise RuntimeError(f"NCCL requires {self.world_size} distinct visible CUDA devices")


def build_pipeline_static_accounting(config: PipelineAccountingConfig) -> dict[str, Any]:
    """Calculate stage-local model state and boundary communication."""

    config.validate()
    stage_records = []
    for stage_index in range(config.world_size):
        stage = TransformerPipelineStage.from_dimensions(
            stage_index=stage_index,
            num_stages=config.world_size,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.model_config.d_model,
            num_layers=config.model_config.num_layers,
            num_heads=config.model_config.num_heads,
            d_ff=config.model_config.d_ff,
            rope_theta=config.rope_theta,
            device=torch.device("meta"),
            dtype=torch.float32,
        )
        parameter_count = sum(parameter.numel() for parameter in stage.parameters())
        parameter_bytes = parameter_count * torch.empty((), dtype=torch.float32).element_size()
        stage_records.append(
            {
                "stage_index": stage_index,
                "layer_start": stage.partition.start,
                "layer_end": stage.partition.end,
                "layer_count": stage.partition.layer_count,
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "persistent_parameter_gradient_moment_bytes": 4 * parameter_bytes,
            }
        )

    activation_elements_per_boundary = config.global_batch_size * config.context_length * config.model_config.d_model
    activation_bytes_per_boundary = activation_elements_per_boundary * torch.empty((), dtype=torch.float32).element_size()
    parameter_bytes_by_stage = [record["parameter_bytes"] for record in stage_records]
    return {
        "world_size": config.world_size,
        "model_config": asdict(config.model_config),
        "vocab_size": config.vocab_size,
        "global_batch_size": config.global_batch_size,
        "context_length": config.context_length,
        "num_microbatches": config.num_microbatches,
        "microbatch_size": config.microbatch_size,
        "dtype": "torch.float32",
        "stages": stage_records,
        "total_parameter_bytes": sum(parameter_bytes_by_stage),
        "max_stage_parameter_bytes": max(parameter_bytes_by_stage),
        "min_stage_parameter_bytes": min(parameter_bytes_by_stage),
        "parameter_imbalance_ratio": max(parameter_bytes_by_stage) / min(parameter_bytes_by_stage),
        "forward_activation_bytes_per_boundary_per_batch": activation_bytes_per_boundary,
        "backward_activation_gradient_bytes_per_boundary_per_batch": activation_bytes_per_boundary,
        "bidirectional_boundary_bytes_per_batch": 2 * activation_bytes_per_boundary,
        "ideal_fill_drain_efficiency": config.num_microbatches / (config.num_microbatches + config.world_size - 1),
    }


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _optimizer_state_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(_tensor_bytes(value) for state in optimizer.state.values() for value in state.values() if isinstance(value, Tensor))


def _read_rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(
    *,
    phase: str,
    stage: TransformerPipelineStage,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    _synchronize(device)
    gradients = tuple(parameter.grad for parameter in stage.parameters() if parameter.grad is not None)
    snapshot: dict[str, Any] = {
        "phase": phase,
        "rss_bytes": _read_rss_bytes(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "local_parameter_bytes": sum(_tensor_bytes(parameter) for parameter in stage.parameters()),
        "local_gradient_bytes": sum(_tensor_bytes(gradient) for gradient in gradients),
        "optimizer_state_tensor_bytes": _optimizer_state_tensor_bytes(optimizer),
    }
    if device.type == "cuda":
        snapshot.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    else:
        snapshot.update(
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
        )
    return snapshot


def _language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _run_step(
    *,
    pipeline: PipelineParallel,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    num_microbatches: int,
    device: torch.device,
    measure: bool,
) -> tuple[float | None, Tensor | None]:
    if measure:
        dist.barrier()
        _synchronize(device)
        started_ns = time.perf_counter_ns()

    optimizer.zero_grad(set_to_none=True)
    loss = pipeline.forward_backward(
        input_ids,
        targets,
        loss_fn=_language_model_loss,
        num_microbatches=num_microbatches,
    )
    optimizer.step()

    if not measure:
        return None, loss
    _synchronize(device)
    dist.barrier()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
    return elapsed_ms, loss


def _worker(
    rank: int,
    config: PipelineAccountingConfig,
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
    try:
        torch.manual_seed(config.seed + rank)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stage = TransformerPipelineStage.from_dimensions(
            stage_index=rank,
            num_stages=config.world_size,
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
        pipeline = PipelineParallel(stage)
        optimizer = AdamW(
            stage.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        memory_snapshots = [
            _memory_snapshot(
                phase="after_stage_initialization",
                stage=stage,
                optimizer=optimizer,
                device=device,
            )
        ]

        generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
        shape = (config.global_batch_size, config.context_length)
        input_ids = torch.randint(config.vocab_size, shape, generator=generator, device=device)
        targets = torch.randint(config.vocab_size, shape, generator=generator, device=device)

        optimizer.zero_grad(set_to_none=True)
        pipeline.forward_backward(
            input_ids,
            targets,
            loss_fn=_language_model_loss,
            num_microbatches=config.num_microbatches,
        )
        memory_snapshots.append(
            _memory_snapshot(
                phase="before_optimizer_step",
                stage=stage,
                optimizer=optimizer,
                device=device,
            )
        )
        optimizer.step()
        memory_snapshots.append(
            _memory_snapshot(
                phase="after_optimizer_step",
                stage=stage,
                optimizer=optimizer,
                device=device,
            )
        )

        for _ in range(config.warmup_steps):
            _run_step(
                pipeline=pipeline,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                num_microbatches=config.num_microbatches,
                device=device,
                measure=False,
            )

        step_samples_ms = []
        loss_samples = []
        for _ in range(config.measurement_steps):
            elapsed_ms, loss = _run_step(
                pipeline=pipeline,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                num_microbatches=config.num_microbatches,
                device=device,
                measure=True,
            )
            if elapsed_ms is None:
                raise RuntimeError("measured pipeline step did not return a duration")
            step_samples_ms.append(elapsed_ms)
            if loss is not None:
                loss_samples.append(loss.item())

        rank_result = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "partition": asdict(stage.partition),
            "parameter_count": sum(parameter.numel() for parameter in stage.parameters()),
            "memory_snapshots": memory_snapshots,
            "step_samples_ms": step_samples_ms,
            "step_summary": summarize_samples(step_samples_ms),
            "loss_samples": loss_samples,
        }
        gathered = [None] * config.world_size if rank == 0 else None
        dist.gather_object(rank_result, gathered, dst=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        if gathered is None or any(result is None for result in gathered):
            raise RuntimeError("rank 0 did not receive every pipeline accounting result")
        rank_results = sorted(
            (result for result in gathered if result is not None),
            key=lambda result: result["rank"],
        )
        rank_max_step_samples_ms = [max(samples) for samples in zip(*(result["step_samples_ms"] for result in rank_results), strict=True)]
        _write_json(
            Path(output_path),
            {
                "status": "passed",
                "config": asdict(config),
                "static_accounting": build_pipeline_static_accounting(config),
                "rank_max_step_samples_ms": rank_max_step_samples_ms,
                "rank_max_step_summary": summarize_samples(rank_max_step_samples_ms),
                "ranks": rank_results,
            },
        )


def run_pipeline_accounting_case(
    config: PipelineAccountingConfig,
    output_path: Path,
) -> dict[str, Any]:
    """Run one isolated distributed pipeline-parallel case."""

    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    wall_started = time.perf_counter()
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
        case_wall_seconds=time.perf_counter() - wall_started,
        environment=_collect_environment(config),
    )
    _write_json(output_path, result)
    return result


def _collect_environment(config: PipelineAccountingConfig) -> dict[str, Any]:
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


def build_xl_pipeline_reference(
    *,
    world_size: int,
    vocab_size: int,
    global_batch_size: int,
    context_length: int,
    num_microbatches: int,
) -> dict[str, Any]:
    """Build an xl PP reference without allocating real model storage."""

    return build_pipeline_static_accounting(
        PipelineAccountingConfig(
            backend="gloo",
            world_size=world_size,
            model_config=MODEL_CONFIGS["xl"],
            vocab_size=vocab_size,
            global_batch_size=global_batch_size,
            context_length=context_length,
            num_microbatches=num_microbatches,
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
