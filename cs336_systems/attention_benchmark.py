"""CPU/GPU timing and memory accounting for naive scaled-dot-product attention."""

from __future__ import annotations

import gc
import os
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.saved_tensor_profiler import capture_saved_tensors


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch_size: int
    sequence_length: int
    d_model: int
    warmup_steps: int
    measurement_steps: int
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True)
class TimingStatistics:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def attention_size_bytes(config: AttentionBenchmarkConfig) -> dict[str, int]:
    element_size = torch.empty((), dtype=torch.float32).element_size()
    activation_bytes = config.batch_size * config.sequence_length * config.d_model * element_size
    score_bytes = config.batch_size * config.sequence_length * config.sequence_length * element_size
    return {
        "one_qkv_or_output_tensor": activation_bytes,
        "qkv_tensors": 3 * activation_bytes,
        "attention_score_tensor": score_bytes,
    }


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss if sys.platform == "darwin" else max_rss * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(samples_seconds: list[float]) -> TimingStatistics:
    samples_ms = [sample * 1_000 for sample in samples_seconds]
    return TimingStatistics(
        mean_ms=statistics.mean(samples_ms),
        std_ms=statistics.pstdev(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def _attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return scaled_dot_product_attention(q, k, v, mask=None)


def benchmark_attention_case(config: AttentionBenchmarkConfig) -> dict[str, Any]:
    if (
        min(
            config.batch_size,
            config.sequence_length,
            config.d_model,
            config.measurement_steps,
        )
        <= 0
    ):
        raise ValueError("batch size, sequence length, d_model, and measurement steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup steps cannot be negative")

    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    rss_before_inputs = _rss_bytes()
    shape = (config.batch_size, config.sequence_length, config.d_model)
    q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    output_gradient = torch.randn(shape, device=device, dtype=torch.float32)
    _synchronize(device)
    rss_after_inputs = _rss_bytes()

    for _ in range(config.warmup_steps):
        output = _attention(q, k, v)
        output.backward(output_gradient)
        q.grad = None
        k.grad = None
        v.grad = None
        del output
    _synchronize(device)
    gc.collect()

    forward_samples: list[float] = []
    for _ in range(config.measurement_steps):
        start = time.perf_counter()
        output = _attention(q, k, v)
        _synchronize(device)
        forward_samples.append(time.perf_counter() - start)
        del output

    gc.collect()
    output, saved_tensor_profile = capture_saved_tensors(
        lambda: _attention(q, k, v),
        tensor_roles={"input": (q, k, v)},
    )
    _synchronize(device)
    rss_before_backward = _rss_bytes()
    peak_rss_before_backward = _peak_rss_bytes()

    backward_samples: list[float] = []
    for _ in range(config.measurement_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        start = time.perf_counter()
        output.backward(output_gradient, retain_graph=True)
        _synchronize(device)
        backward_samples.append(time.perf_counter() - start)

    saved_metrics = saved_tensor_profile.metrics()
    result = {
        "status": "ok",
        "config": {
            **asdict(config),
            "dtype": "float32",
            "torch_version": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
        },
        "theoretical_bytes": attention_size_bytes(config),
        "timings": {
            "forward": asdict(_summarize(forward_samples)),
            "backward": asdict(_summarize(backward_samples)),
        },
        "memory": {
            "rss_before_inputs": rss_before_inputs,
            "rss_after_inputs": rss_after_inputs,
            "rss_before_backward": rss_before_backward,
            "peak_rss_before_backward": peak_rss_before_backward,
            "peak_rss_after_backward": _peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "cuda_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        },
        "saved_tensors_after_forward": {
            "reference_count": saved_metrics.tensor_count,
            "logical_bytes": saved_metrics.logical_bytes,
            "unique_storage_bytes": saved_metrics.unique_storage_bytes,
        },
        "numerics": {
            "output_sum": float(output.detach().double().sum()),
            "q_gradient_norm": None if q.grad is None else float(q.grad.detach().double().norm()),
            "k_gradient_norm": None if k.grad is None else float(k.grad.detach().double().norm()),
            "v_gradient_norm": None if v.grad is None else float(v.grad.detach().double().norm()),
        },
    }
    return result
