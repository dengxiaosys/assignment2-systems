"""Shared input validation for all attention implementations."""

import torch


def validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if any(tensor.ndim != 2 for tensor in (q, k, v)):
        raise ValueError("Q/K/V must each be a 2D tensor of shape (S, d)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must have the same shape (S, d)")
    if min(q.shape) <= 0:
        raise ValueError("S and d must be positive")
    if q.device != k.device or k.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("Q/K/V must have the same dtype")
    if q.dtype != torch.float32:
        raise TypeError("Q/K/V must use float32")
