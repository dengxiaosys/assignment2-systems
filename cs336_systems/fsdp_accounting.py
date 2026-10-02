"""Static memory analysis and distributed runtime accounting for FSDP."""

from __future__ import annotations

import json
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
from torch import Tensor, nn
from torch.cuda import nccl

from cs336_basics.model import Embedding, Linear, TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig, validate_config
from cs336_systems.fsdp import FullyShardedDataParallel
from cs336_systems.fsdp_observer import FSDPAllGatherEvent
from cs336_systems.optimizer_sharding_accounting import summarize_samples

_SHARDABLE_MODULE_TYPES = (Linear, Embedding, nn.Linear, nn.Embedding)


@dataclass(frozen=True)
class FSDPAccountingConfig:
    """Configuration for one isolated FSDP accounting run."""

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
    compute_dtype: torch.dtype | None = None
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
        if self.compute_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported compute_dtype")
        if self.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("Gloo is unavailable")
        if self.backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < self.world_size):
            raise RuntimeError(f"NCCL requires {self.world_size} distinct visible CUDA devices")


@dataclass
class _PendingAllGatherMeasurement:
    event: FSDPAllGatherEvent
    started_ns: int
    wait_started_ns: int | None = None
    ready_before_wait: bool | None = None


class FSDPCommunicationRecorder:
    """Collect all-gather timing from the optional FSDP observer seam."""

    def __init__(self) -> None:
        self._pending: dict[int, _PendingAllGatherMeasurement] = {}
        self._records: list[dict[str, Any]] = []

    def __call__(self, event: FSDPAllGatherEvent) -> None:
        work_id = id(event.work)
        if event.stage == "launched":
            if work_id in self._pending:
                raise RuntimeError("received a duplicate all-gather launch event")
            self._pending[work_id] = _PendingAllGatherMeasurement(
                event=event,
                started_ns=time.perf_counter_ns(),
            )
            return

        measurement = self._pending.get(work_id)
        if measurement is None:
            raise RuntimeError(f"received an all-gather {event.stage} event before launch")
        if event.stage == "waiting":
            measurement.ready_before_wait = event.work.is_completed()
            measurement.wait_started_ns = time.perf_counter_ns()
            if event.device.type == "cuda":
                torch.cuda.nvtx.range_push(f"fsdp_all_gather_wait:{event.module_name}")
            return

        if measurement.wait_started_ns is None or measurement.ready_before_wait is None:
            raise RuntimeError("received an all-gather finished event before waiting")
        finished_ns = time.perf_counter_ns()
        if event.device.type == "cuda":
            torch.cuda.nvtx.range_pop()
        self._records.append(
            {
                "phase": measurement.event.phase,
                "module_name": measurement.event.module_name,
                "communicated_dtype": str(measurement.event.communicated_dtype),
                "communicated_full_bytes": measurement.event.communicated_numel * torch.empty((), dtype=measurement.event.communicated_dtype).element_size(),
                "total_ms": (finished_ns - measurement.started_ns) / 1e6,
                "wait_ms": (finished_ns - measurement.wait_started_ns) / 1e6,
                "ready_before_wait": measurement.ready_before_wait,
            }
        )
        del self._pending[work_id]

    def snapshot(self, *, reset: bool = False) -> dict[str, Any]:
        """Return completed records, optionally clearing them."""

        if self._pending:
            pending_modules = sorted(measurement.event.module_name for measurement in self._pending.values())
            raise RuntimeError(f"cannot snapshot while all-gather operations are pending: {pending_modules}")
        records = list(self._records)
        snapshot = {
            "all_gather_calls": len(records),
            "all_gather_total_ms": [record["total_ms"] for record in records],
            "all_gather_wait_ms": [record["wait_ms"] for record in records],
            "all_gather_wait_total_ms": sum(record["wait_ms"] for record in records),
            "all_gather_records": records,
        }
        if reset:
            self._records.clear()
        return snapshot


def build_fsdp_static_accounting(
    *,
    model_config: ModelConfig,
    vocab_size: int,
    context_length: int,
    world_size: int,
) -> dict[str, Any]:
    """Calculate persistent FP32 state with shardable and replicated parameters."""

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
    shardable_ids = {id(child.weight) for child in model.modules() if isinstance(child, _SHARDABLE_MODULE_TYPES)}
    shardable_parameters = tuple(parameter for parameter in parameters if id(parameter) in shardable_ids)
    replicated_parameters = tuple(parameter for parameter in parameters if id(parameter) not in shardable_ids)

    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in parameters)
    shardable_full_bytes = sum(_tensor_bytes(parameter) for parameter in shardable_parameters)
    replicated_bytes = sum(_tensor_bytes(parameter) for parameter in replicated_parameters)
    sharded_storage_bytes_per_rank = sum(((parameter.numel() + world_size - 1) // world_size) * parameter.element_size() for parameter in shardable_parameters)
    fsdp_local_parameter_bytes = sharded_storage_bytes_per_rank + replicated_bytes

    rank_to_optimizer_owner_bytes = [0] * world_size
    for parameter in parameters:
        owner_rank = min(
            range(world_size),
            key=lambda rank: (rank_to_optimizer_owner_bytes[rank], rank),
        )
        rank_to_optimizer_owner_bytes[owner_rank] += _tensor_bytes(parameter)
    optimizer_sharding_persistent_by_rank = [2 * parameter_bytes + 2 * owner_bytes for owner_bytes in rank_to_optimizer_owner_bytes]
    fsdp_persistent_bytes_per_rank = 4 * fsdp_local_parameter_bytes

    return {
        "model_config": asdict(model_config),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "world_size": world_size,
        "dtype": "torch.float32",
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_tensors": len(parameters),
        "shardable_parameter_tensors": len(shardable_parameters),
        "replicated_parameter_tensors": len(replicated_parameters),
        "full_parameter_bytes": parameter_bytes,
        "shardable_full_parameter_bytes": shardable_full_bytes,
        "replicated_parameter_bytes": replicated_bytes,
        "sharded_storage_bytes_per_rank_including_padding": sharded_storage_bytes_per_rank,
        "fsdp_local_parameter_bytes_per_rank": fsdp_local_parameter_bytes,
        "fsdp_local_gradient_bytes_per_rank": fsdp_local_parameter_bytes,
        "fsdp_local_adamw_moment_bytes_per_rank": 2 * fsdp_local_parameter_bytes,
        "fsdp_persistent_parameter_gradient_moment_bytes_per_rank": fsdp_persistent_bytes_per_rank,
        "optimizer_sharding_persistent_bytes_by_rank": optimizer_sharding_persistent_by_rank,
        "peak_savings_vs_optimizer_sharding_rank_max_bytes": max(optimizer_sharding_persistent_by_rank) - fsdp_persistent_bytes_per_rank,
        "peak_savings_vs_optimizer_sharding_rank_max_percent": 100
        * (max(optimizer_sharding_persistent_by_rank) - fsdp_persistent_bytes_per_rank)
        / max(optimizer_sharding_persistent_by_rank),
        "persistent_savings_vs_replicated_ddp_bytes": 4 * parameter_bytes - fsdp_persistent_bytes_per_rank,
        "persistent_savings_vs_replicated_ddp_percent": 100 * (4 * parameter_bytes - fsdp_persistent_bytes_per_rank) / (4 * parameter_bytes),
    }


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _read_rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _optimizer_state_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(_tensor_bytes(value) for state in optimizer.state.values() for value in state.values() if isinstance(value, Tensor))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(
    *,
    phase: str,
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    _synchronize(device)
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
    snapshot: dict[str, Any] = {
        "phase": phase,
        "rss_bytes": _read_rss_bytes(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "local_parameter_bytes": sum(_tensor_bytes(parameter) for parameter in parameters),
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


def _run_step(
    *,
    model: FullyShardedDataParallel,
    communication_recorder: FSDPCommunicationRecorder,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    measure: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    if measure:
        dist.barrier()
        _synchronize(device)
        communication_recorder.snapshot(reset=True)
        step_started_ns = time.perf_counter_ns()

    optimizer.zero_grad(set_to_none=True)
    forward_started_ns = time.perf_counter_ns()
    logits = model(input_ids)
    _synchronize(device)
    forward_ms = (time.perf_counter_ns() - forward_started_ns) / 1e6

    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    model.finish_gradient_synchronization()
    _synchronize(device)
    optimizer.step()
    _synchronize(device)

    if not measure:
        return {}, {}
    timings = {
        "step_total_ms": (time.perf_counter_ns() - step_started_ns) / 1e6,
        "forward_ms": forward_ms,
    }
    return timings, communication_recorder.snapshot()


def _worker(
    rank: int,
    config: FSDPAccountingConfig,
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
        module = TransformerLM(
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
        communication_recorder = FSDPCommunicationRecorder()
        model = FullyShardedDataParallel(
            module,
            compute_dtype=config.compute_dtype,
            observer=communication_recorder,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        memory_snapshots = [
            _memory_snapshot(
                phase="after_fsdp_initialization",
                model=model,
                optimizer=optimizer,
                device=device,
            )
        ]

        generator = torch.Generator(device=device).manual_seed(config.seed + 10_000 + rank)
        shape = (config.local_batch_size, config.context_length)
        input_ids = torch.randint(config.vocab_size, shape, generator=generator, device=device)
        targets = torch.randint(config.vocab_size, shape, generator=generator, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        model.finish_gradient_synchronization()
        memory_snapshots.append(
            _memory_snapshot(
                phase="before_optimizer_step",
                model=model,
                optimizer=optimizer,
                device=device,
            )
        )
        optimizer.step()
        memory_snapshots.append(
            _memory_snapshot(
                phase="after_optimizer_step",
                model=model,
                optimizer=optimizer,
                device=device,
            )
        )

        for _ in range(config.warmup_steps):
            _run_step(
                model=model,
                communication_recorder=communication_recorder,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
                measure=False,
            )

        timing_samples = {"step_total_ms": [], "forward_ms": []}
        communication_samples = []
        for _ in range(config.measurement_steps):
            timings, communication = _run_step(
                model=model,
                communication_recorder=communication_recorder,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                device=device,
                measure=True,
            )
            for name, value in timings.items():
                timing_samples[name].append(value)
            communication_samples.append(communication)

        rank_result = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "memory_snapshots": memory_snapshots,
            "timing_samples_ms": timing_samples,
            "timing_summaries": {name: summarize_samples(values) for name, values in timing_samples.items()},
            "communication_samples": communication_samples,
            "communication_summary": _summarize_communication(communication_samples),
        }
        gathered = [None] * config.world_size if rank == 0 else None
        dist.gather_object(rank_result, gathered, dst=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        if gathered is None or any(result is None for result in gathered):
            raise RuntimeError("rank 0 did not receive every FSDP accounting result")
        rank_results = sorted(
            (result for result in gathered if result is not None),
            key=lambda result: result["rank"],
        )
        _write_json(
            Path(output_path),
            _build_result(config, rank_results),
        )


def _summarize_communication(samples: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for sample in samples for record in sample["all_gather_records"]]
    summary = {
        "measurement_steps": len(samples),
        "all_gather_calls": len(records),
        "forward": _summarize_phase_records(records, "forward"),
        "backward": _summarize_phase_records(records, "backward"),
    }
    return summary


def _summarize_phase_records(records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matching = [record for record in records if record["phase"] == phase]
    return {
        "calls": len(matching),
        "communicated_full_bytes": sum(record["communicated_full_bytes"] for record in matching),
        "ready_before_wait_count": sum(record["ready_before_wait"] for record in matching),
        "ready_before_wait_fraction": (sum(record["ready_before_wait"] for record in matching) / len(matching) if matching else None),
        "wait_total_ms": sum(record["wait_ms"] for record in matching),
        "wait_mean_ms": statistics.fmean(record["wait_ms"] for record in matching) if matching else None,
        "all_gather_total_mean_ms": statistics.fmean(record["total_ms"] for record in matching) if matching else None,
    }


def run_fsdp_accounting_case(
    config: FSDPAccountingConfig,
    output_path: Path,
) -> dict[str, Any]:
    """Run one distributed FSDP accounting case."""

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


def _build_result(
    config: FSDPAccountingConfig,
    rank_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if [result["rank"] for result in rank_results] != list(range(config.world_size)):
        raise ValueError("rank results are missing or duplicated")
    rank_max_samples = {}
    for name in ("step_total_ms", "forward_ms"):
        per_rank = [result["timing_samples_ms"][name] for result in rank_results]
        rank_max_samples[name] = [max(values) for values in zip(*per_rank, strict=True)]
    return {
        "status": "passed",
        "config": {
            **asdict(config),
            "compute_dtype": str(config.compute_dtype) if config.compute_dtype is not None else None,
        },
        "rank_max_timing_samples_ms": rank_max_samples,
        "rank_max_timing_summaries": {name: summarize_samples(values) for name, values in rank_max_samples.items()},
        "ranks": rank_results,
    }


def _collect_environment(config: FSDPAccountingConfig) -> dict[str, Any]:
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


def build_xl_fsdp_reference(
    *,
    vocab_size: int,
    context_length: int,
    world_size: int,
) -> dict[str, Any]:
    return build_fsdp_static_accounting(
        model_config=MODEL_CONFIGS["xl"],
        vocab_size=vocab_size,
        context_length=context_length,
        world_size=world_size,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
