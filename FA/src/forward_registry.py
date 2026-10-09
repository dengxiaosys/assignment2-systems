"""Resolve and validate named FP32 attention forward implementations."""

from collections.abc import Callable

import torch

from .cuda_fa2_forward import (
    cuda_fa2_attention_forward,
    validate_cuda_fa2_support,
)
from .input_validation import validate_attention_inputs
from .memory_efficient_forward import (
    memory_efficient_attention_forward,
    validate_memory_efficient_support,
)
from .native_forward import native_attention_forward


IMPLEMENTATION_BACKENDS = {
    "native": "pytorch_dense_attention",
    "efficient": "pytorch_sdpa_efficient_attention",
    "cuda_fa2": "custom_cpp_cuda_fa2_baseline_fp32",
}


def get_attention_forward(implementation: str) -> Callable[..., torch.Tensor]:
    """Resolve once, then reuse the callable for verification, warmup and capture."""
    if implementation == "native":
        return native_attention_forward
    elif implementation == "efficient":
        return memory_efficient_attention_forward
    elif implementation == "cuda_fa2":
        return cuda_fa2_attention_forward
    else:
        raise ValueError(
            f"Unknown implementation {implementation!r}; "
            f"choose {tuple(IMPLEMENTATION_BACKENDS)}"
        )


def validate_implementation(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> None:
    """Validate common inputs and implementation-specific runtime support."""
    if implementation == "native":
        validate_attention_inputs(q, k, v)
        return
    elif implementation == "efficient":
        validate_memory_efficient_support(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    elif implementation == "cuda_fa2":
        validate_cuda_fa2_support(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    else:
        raise ValueError(f"Unknown implementation {implementation!r}")
