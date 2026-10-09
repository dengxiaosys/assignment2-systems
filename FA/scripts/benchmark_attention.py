"""Benchmark native, memory-efficient SDPA or custom CUDA FA2 on FP32 inputs."""

import argparse
import json
from pathlib import Path

import torch

from src.benchmark_measurement import measure_latency, measure_peak_bytes
from src.forward_registry import (
    IMPLEMENTATION_BACKENDS,
    get_attention_forward,
    validate_implementation,
)
from src.numerical_verification import (
    REFERENCE_TOLERANCE,
    verify_against_cpu_fp64,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options and validate their values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--impl", choices=tuple(IMPLEMENTATION_BACKENDS), default="native")
    parser.add_argument("--seq-len", type=int, default=16384, help="Sequence length S")
    parser.add_argument("--head-dim", type=int, default=64, help="Feature dimension d")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true", help="Skip the CPU FP64 reference for large cases")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if min(args.seq_len, args.head_dim, args.iterations, args.repeats, args.threads) <= 0:
        parser.error("Dimensions, iterations, repeats and threads must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.device == "cpu" and args.impl != "native":
        parser.error("efficient and cuda_fa2 require --device cuda")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check GPU permissions and the selected environment (no CPU fallback)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    forward = get_attention_forward(args.impl)
    q = torch.randn(args.seq_len, args.head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    validate_implementation(args.impl, q, k, v, is_causal=args.causal)
    error = None if args.no_verify else verify_against_cpu_fp64(
        forward,
        q,
        k,
        v,
        is_causal=args.causal,
    )

    def operation() -> torch.Tensor:
        return forward(q, k, v, is_causal=args.causal)

    timing = measure_latency(operation, device, args.warmup, args.iterations, args.repeats)
    peak = measure_peak_bytes(operation, device)
    matrix_bytes = args.seq_len**2 * q.element_size()
    payload = {
        "implementation": args.impl,
        "backend": IMPLEMENTATION_BACKENDS[args.impl],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "dtype": str(q.dtype),
        "tf32": False,
        "qkv_shape": list(q.shape),
        "causal": args.causal,
        "cpu_threads": args.threads,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "verified": not args.no_verify,
        "verification_rtol": REFERENCE_TOLERANCE if not args.no_verify else None,
        "verification_atol": REFERENCE_TOLERANCE if not args.no_verify else None,
        "max_abs_error_vs_cpu_fp64": error,
        "materializes_full_attention_matrices": args.impl == "native",
        "native_s_matrix_bytes": matrix_bytes,
        "native_p_matrix_bytes": matrix_bytes,
        "native_s_plus_p_bytes": 2 * matrix_bytes,
        "peak_additional_cuda_allocated_bytes": peak,
        **timing,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
