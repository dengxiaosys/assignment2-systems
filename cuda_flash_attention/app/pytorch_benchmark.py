#!/usr/bin/env python3

import argparse
import gc
import json
import math
import warnings
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
_CAUSAL_MASKS: dict[tuple[torch.device, int, int], torch.Tensor] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention on one CUDA device.")
    parser.add_argument("--nq", type=int, required=True)
    parser.add_argument("--nk", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.nq <= 0 or args.nk <= 0 or args.head_dim <= 0:
        raise ValueError("nq, nk, and head-dim must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def cuda_time_ms(operation: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def default_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, is_causal=causal)


def math_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(query, key, value, is_causal=causal)


def efficient_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(query, key, value, is_causal=causal)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if causal:
        query_count = query.shape[-2]
        key_count = key.shape[-2]
        mask_key = (query.device, query_count, key_count)
        visible = _CAUSAL_MASKS.get(mask_key)
        if visible is None:
            visible = torch.ones((query_count, key_count), dtype=torch.bool, device=query.device).tril()
            _CAUSAL_MASKS[mask_key] = visible
        scores = scores.masked_fill(~visible, -torch.inf)
    return torch.matmul(torch.softmax(scores, dim=-1), value)


def measure_peak_bytes(
    attention: AttentionFunction,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()

    measured_query = query.detach().clone().requires_grad_(True)
    measured_key = key.detach().clone().requires_grad_(True)
    measured_value = value.detach().clone().requires_grad_(True)
    output = attention(measured_query, measured_key, measured_value, causal)
    torch.autograd.grad(output, (measured_query, measured_key, measured_value), grad_output)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline_bytes


def benchmark_implementation(
    name: str,
    attention: AttentionFunction,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    with torch.no_grad():
        forward_ms = cuda_time_ms(
            lambda: attention(query, key, value, causal),
            warmup,
            iterations,
        )

    backward_query = query.detach().clone().requires_grad_(True)
    backward_key = key.detach().clone().requires_grad_(True)
    backward_value = value.detach().clone().requires_grad_(True)
    backward_output = attention(backward_query, backward_key, backward_value, causal)

    def run_backward() -> None:
        torch.autograd.grad(
            backward_output,
            (backward_query, backward_key, backward_value),
            grad_output,
            retain_graph=True,
        )

    backward_ms = cuda_time_ms(run_backward, warmup, iterations)
    peak_bytes = measure_peak_bytes(attention, query, key, value, grad_output, causal)
    return {
        "implementation": name,
        "forward_mean_ms": forward_ms,
        "backward_mean_ms": backward_ms,
        "peak_additional_allocated_bytes": peak_bytes,
    }


def probe_backend(
    backend: SDPBackend,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> dict[str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(query, key, value, is_causal=causal)
        torch.cuda.synchronize()
        return {"status": "available"}
    except RuntimeError as error:
        return {"status": "unavailable", "reason": str(error).splitlines()[0]}


def max_abs_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    shape_query = (1, 1, args.nq, args.head_dim)
    shape_key = (1, 1, args.nk, args.head_dim)
    query = torch.empty(shape_query, device=device).uniform_(-0.5, 0.5)
    key = torch.empty(shape_key, device=device).uniform_(-0.5, 0.5)
    value = torch.empty(shape_key, device=device).uniform_(-0.5, 0.5)
    grad_output = torch.empty(shape_query, device=device).uniform_(-0.5, 0.5)

    with torch.no_grad():
        expected = eager_attention(query, key, value, args.causal)
        default_output = default_sdpa(query, key, value, args.causal)
        math_output = math_sdpa(query, key, value, args.causal)

    backend_probe = {}
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ):
        backend_probe[name] = probe_backend(backend, query, key, value, args.causal)

    implementations = (
        ("pytorch_sdpa_default", default_sdpa),
        ("pytorch_sdpa_efficient", efficient_sdpa),
        ("pytorch_sdpa_math", math_sdpa),
        ("pytorch_eager", eager_attention),
    )
    results = [
        benchmark_implementation(
            name,
            attention,
            query,
            key,
            value,
            grad_output,
            args.causal,
            args.warmup,
            args.iterations,
        )
        for name, attention in implementations
    ]

    payload = {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "query_count": args.nq,
        "key_count": args.nk,
        "head_dim": args.head_dim,
        "causal": args.causal,
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "default_vs_eager_max_abs_error": max_abs_difference(default_output, expected),
        "math_vs_eager_max_abs_error": max_abs_difference(math_output, expected),
        "backend_probe": backend_probe,
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
        print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
