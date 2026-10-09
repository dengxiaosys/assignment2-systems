"""Reusable latency and peak-allocation measurements."""

import statistics
import time
from collections.abc import Callable

import torch


def measure_latency(
    operation: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    """Measure per-call wall time and CUDA Event time across repeated batches."""
    for _ in range(warmup):
        operation()

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    wall_samples = []
    event_samples = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        if cuda:
            start.record()
        for _ in range(iterations):
            operation()
        if cuda:
            end.record()
            end.synchronize()
        wall_samples.append((time.perf_counter() - begin) * 1000 / iterations)
        if cuda:
            event_samples.append(start.elapsed_time(end) / iterations)

    return {
        "wall_ms_mean": statistics.mean(wall_samples),
        "wall_ms_median": statistics.median(wall_samples),
        "wall_ms_samples": wall_samples,
        "cuda_event_ms_mean": statistics.mean(event_samples) if cuda else None,
        "cuda_event_ms_median": statistics.median(event_samples) if cuda else None,
        "cuda_event_ms_samples": event_samples,
    }


def measure_peak_bytes(
    operation: Callable[[], torch.Tensor],
    device: torch.device,
) -> int | None:
    """Measure live PyTorch allocations added above the pre-existing inputs."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = operation()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del output
    return peak
