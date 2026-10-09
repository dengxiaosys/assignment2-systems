"""Step 0: single-head attention on (S, d), materializing full S-by-S matrices."""

import torch


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
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


@torch.no_grad()
def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Return O = softmax(Q @ K.T / sqrt(d)) @ V for single-head self-attention.

    Q/K/V/O: (S, d). Scores and probabilities: (S, S).
    Both full intermediate matrices remain alive during P @ V.
    Causal visibility is key_index <= query_index.
    Only forward is implemented; autograd is disabled. Inputs and scaled
    dot products must be finite. Run outside mixed-precision autocast.
    """
    _validate_inputs(q, k, v)
    seq_len, d = q.shape

    # 1. Materialize the entire score matrix: (S, S).
    s = q @ k.T / (d**0.5)
    if is_causal:
        query_index = torch.arange(seq_len, device=q.device)[:, None]
        key_index = torch.arange(seq_len, device=q.device)[None, :]
        s = s.masked_fill(key_index > query_index, -torch.inf)

    # 2. Materialize a second matrix of the same size for probabilities.
    p = torch.softmax(s, dim=1)

    # 3. Both S and P are still live here; no tiling or kernel fusion.
    o = p @ v
    return o
