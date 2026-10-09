"""Select an FP32 forward implementation with the public (S, d) interface."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .forward import _validate_inputs, attention_forward


BACKEND_NAMES = {
    "native": "pytorch_dense_attention",
    "efficient": "pytorch_sdpa_efficient_attention",
}


def _require_cuda(q: torch.Tensor) -> None:
    if q.device.type != "cuda":
        raise RuntimeError("efficient requires CUDA tensors; use --impl native on CPU")


@torch.no_grad()
def efficient_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Force FP32 memory-efficient SDPA; unsupported inputs raise without fallback."""
    _validate_inputs(q, k, v)
    _require_cuda(q)
    # Singleton batch/head axes are views, with no copy or dtype conversion.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None],
            dropout_p=0.0, is_causal=is_causal,
        )
    return output[0, 0]


def get_attention_kernel(implementation: str) -> Callable[..., torch.Tensor]:
    """Resolve once, then reuse the callable for verification, warmup and capture."""
    if implementation == "native":
        return attention_forward
    elif implementation == "efficient":
        return efficient_attention_forward
    else:
        raise ValueError(f"Unknown implementation {implementation!r}; choose {tuple(BACKEND_NAMES)}")


@torch.no_grad()
def validate_backend(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> None:
    """Check installed-build/device/input support before warmup or profiling."""
    _validate_inputs(q, k, v)
    if implementation == "native":
        return
    if implementation != "efficient":
        raise ValueError(f"Unknown implementation {implementation!r}")

    _require_cuda(q)
    params = torch.backends.cuda.SDPAParams(
        q[None, None], k[None, None], v[None, None],
        None, 0.0, is_causal, False,
    )
    # PyTorch reports detailed shape/alignment/architecture restrictions via warnings.
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
