"""PyTorch memory-efficient SDPA forward for FP32 (S, d) inputs."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .input_validation import validate_attention_inputs


def _require_cuda(q: torch.Tensor) -> None:
    if q.device.type != "cuda":
        raise RuntimeError("memory-efficient attention requires CUDA tensors")


def _as_sdpa_input(tensor: torch.Tensor) -> torch.Tensor:
    """View (S, d) as (1, 1, S, d) without copying data."""
    return tensor.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def memory_efficient_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run memory-efficient SDPA without backend fallback or dtype conversion."""
    validate_attention_inputs(q, k, v)
    _require_cuda(q)
    q_sdpa = _as_sdpa_input(q)
    k_sdpa = _as_sdpa_input(k)
    v_sdpa = _as_sdpa_input(v)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    return output[0, 0]


@torch.no_grad()
def validate_memory_efficient_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> None:
    """Check whether the installed PyTorch build can run this fused backend."""
    validate_attention_inputs(q, k, v)
    _require_cuda(q)
    q_sdpa = _as_sdpa_input(q)
    k_sdpa = _as_sdpa_input(k)
    v_sdpa = _as_sdpa_input(v)
    params = torch.backends.cuda.SDPAParams(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        None,
        0.0,
        is_causal,
        False,
    )
    # PyTorch emits detailed architecture/alignment restrictions in debug mode.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        supported = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
    if not supported:
        capability = torch.cuda.get_device_capability(q.device)
        raise RuntimeError(
            "FP32 memory-efficient SDPA is unavailable for this CUDA build, GPU or layout. "
            f"GPU={torch.cuda.get_device_name(q.device)}, capability={capability}, "
            f"dtype={q.dtype}, shape={tuple(q.shape)}, stride={q.stride()}, "
            f"torch={torch.__version__}. No backend fallback or dtype conversion."
        )
