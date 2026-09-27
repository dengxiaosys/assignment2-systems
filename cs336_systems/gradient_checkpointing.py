"""Checkpointing strategies for a sequential stack of PyTorch modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

type CheckpointStrategy = Literal["none", "grouped", "recursive"]
type BlockSequence = Sequence[nn.Module] | nn.ModuleList


def apply_blocks(blocks: BlockSequence, x: Tensor) -> Tensor:
    """Apply all blocks sequentially without introducing checkpoints."""

    for block in blocks:
        x = block(x)
    return x


def apply_grouped_checkpointing(
    blocks: BlockSequence,
    x: Tensor,
    *,
    blocks_per_checkpoint: int,
) -> Tensor:
    """Checkpoint non-overlapping groups without nesting checkpoint calls."""

    if blocks_per_checkpoint <= 0:
        raise ValueError("blocks per checkpoint must be positive")
    for start in range(0, len(blocks), blocks_per_checkpoint):
        stop = min(start + blocks_per_checkpoint, len(blocks))
        segment = tuple(blocks[index] for index in range(start, stop))
        x = checkpoint(
            lambda value, segment=segment: apply_blocks(segment, value),
            x,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return x


def apply_recursive_checkpointing(blocks: BlockSequence, x: Tensor) -> Tensor:
    """Recursively bisect a block stack and checkpoint both child ranges."""

    if not blocks:
        return x
    if len(blocks) == 1:
        return blocks[0](x)

    midpoint = len(blocks) // 2
    left = tuple(blocks[index] for index in range(midpoint))
    right = tuple(blocks[index] for index in range(midpoint, len(blocks)))
    x = checkpoint(
        lambda value: apply_recursive_checkpointing(left, value),
        x,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    return checkpoint(
        lambda value: apply_recursive_checkpointing(right, value),
        x,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def apply_checkpoint_strategy(
    blocks: BlockSequence,
    x: Tensor,
    *,
    strategy: CheckpointStrategy,
    blocks_per_checkpoint: int | None = None,
) -> Tensor:
    """Apply a block stack using the selected checkpointing strategy."""

    if strategy == "none":
        if blocks_per_checkpoint is not None:
            raise ValueError("blocks per checkpoint is only valid for grouped checkpointing")
        return apply_blocks(blocks, x)
    if strategy == "grouped":
        if blocks_per_checkpoint is None:
            raise ValueError("grouped checkpointing requires blocks per checkpoint")
        return apply_grouped_checkpointing(
            blocks,
            x,
            blocks_per_checkpoint=blocks_per_checkpoint,
        )
    if strategy == "recursive":
        if blocks_per_checkpoint is not None:
            raise ValueError("blocks per checkpoint is not used by recursive checkpointing")
        return apply_recursive_checkpointing(blocks, x)
    raise ValueError(f"unsupported checkpoint strategy: {strategy}")
