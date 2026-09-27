import copy

import pytest
import torch
from torch import nn

from cs336_systems.gradient_checkpointing import CheckpointStrategy, apply_checkpoint_strategy


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.silu(self.linear(x))


def _run(
    blocks: nn.ModuleList,
    input_data: torch.Tensor,
    *,
    strategy: CheckpointStrategy,
    blocks_per_checkpoint: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    x = input_data.detach().clone().requires_grad_(True)
    output = apply_checkpoint_strategy(
        blocks,
        x,
        strategy=strategy,
        blocks_per_checkpoint=blocks_per_checkpoint,
    )
    output.square().mean().backward()
    assert x.grad is not None
    parameter_grads = [parameter.grad.detach().clone() for parameter in blocks.parameters() if parameter.grad is not None]
    return output.detach(), x.grad.detach(), parameter_grads


@pytest.mark.parametrize(
    ("strategy", "blocks_per_checkpoint"),
    [
        ("grouped", 2),
        ("recursive", None),
    ],
)
def test_checkpoint_strategies_match_uncheckpointed(
    strategy: CheckpointStrategy,
    blocks_per_checkpoint: int | None,
) -> None:
    torch.manual_seed(0)
    baseline_blocks = nn.ModuleList([ResidualBlock(8) for _ in range(4)])
    checkpointed_blocks = copy.deepcopy(baseline_blocks)
    input_data = torch.randn(2, 3, 8)

    expected = _run(baseline_blocks, input_data, strategy="none")
    actual = _run(
        checkpointed_blocks,
        input_data,
        strategy=strategy,
        blocks_per_checkpoint=blocks_per_checkpoint,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert len(actual[2]) == len(expected[2])
    for actual_grad, expected_grad in zip(actual[2], expected[2], strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_grouped_checkpointing_requires_positive_group_size() -> None:
    blocks = nn.ModuleList([ResidualBlock(4)])
    x = torch.randn(2, 4, requires_grad=True)

    with pytest.raises(ValueError, match="positive"):
        apply_checkpoint_strategy(
            blocks,
            x,
            strategy="grouped",
            blocks_per_checkpoint=0,
        )
