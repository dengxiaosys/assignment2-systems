"""Numerical verification shared by benchmarks and tests."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


REFERENCE_TOLERANCE = 2e-5


def reference_row(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qi: int, causal: bool) -> torch.Tensor:
    """Compute one CPU FP64 attention row without an S x S intermediate."""
    key_limit = qi + 1 if causal else q.shape[0]
    scores = (k[:key_limit] @ q[qi]) / q.shape[1] ** 0.5
    return scores.softmax(0) @ v[:key_limit]


@torch.no_grad()
def verify_against_cpu_fp64(
    kernel: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> float:
    """Check an FP32 implementation against CPU FP64 math SDPA."""
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.cpu().double(),
            k.cpu().double(),
            v.cpu().double(),
            dropout_p=0.0,
            is_causal=is_causal,
        )
    actual = kernel(q, k, v, is_causal=is_causal).cpu().double()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=REFERENCE_TOLERANCE,
        atol=REFERENCE_TOLERANCE,
    )
    return (actual - expected).abs().max().item()
