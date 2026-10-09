"""Step 0: single-head attention on (S, d), materializing full S-by-S matrices."""

import torch

from .input_validation import validate_attention_inputs


@torch.no_grad()
def native_attention_forward(
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
    validate_attention_inputs(q, k, v)
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
