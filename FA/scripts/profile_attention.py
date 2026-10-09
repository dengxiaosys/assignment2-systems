"""Capture exactly one FP32 (S, d) forward call under Nsight Systems or nvprof.

Use nsys --capture-range=cudaProfilerApi or nvprof --profile-from-start off.
This entry point emits NVTX operator ranges; use scripts.benchmark_attention for latency.
"""

import argparse
import json
import sys

import torch
import torch.cuda.profiler

from src.forward_registry import (
    IMPLEMENTATION_BACKENDS,
    get_attention_forward,
    validate_implementation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options and validate their values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=tuple(IMPLEMENTATION_BACKENDS), default="native")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.seq_len, args.head_dim, args.threads) <= 0:
        parser.error("Dimensions and threads must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable: python={sys.executable}, "
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
        )

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    forward = get_attention_forward(args.impl)
    q = torch.randn(args.seq_len, args.head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    validate_implementation(args.impl, q, k, v, is_causal=args.causal)
    print(json.dumps({
        "implementation": args.impl,
        "backend": IMPLEMENTATION_BACKENDS[args.impl],
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "qkv_shape": list(q.shape),
        "dtype": str(q.dtype),
        "causal": args.causal,
        "warmup": args.warmup,
        "captured_iterations": 1,
    }, indent=2), flush=True)

    # Input generation, CUDA/cuBLAS initialization and warmup are outside capture.
    for _ in range(args.warmup):
        forward(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()

    # Annotate the selected forward's ATen operators without changing its algorithm.
    # NVTX ranges measure CPU launch scopes, not GPU kernel execution durations.
    with torch.autograd.profiler.emit_nvtx(record_shapes=False):
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"attention_forward/{args.impl}/float32"):
                forward(q, k, v, is_causal=args.causal)
            # Wait once at the capture boundary, not between kernels.
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()

    print("Capture finished: one forward call. Inspect the external profiler report.")


if __name__ == "__main__":
    main()
