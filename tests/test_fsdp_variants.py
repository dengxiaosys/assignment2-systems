from __future__ import annotations

from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.model import Embedding, Linear, RMSNorm
from cs336_systems.fsdp import FullyShardedDataParallel

from .common import _cleanup_process_group, _setup_process_group


class OddSizedModel(nn.Module):
    """Use non-divisible weights to exercise FSDP padding."""

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(7, 5)
        self.norm = RMSNorm(5)
        self.projection = Linear(5, 3)

    def forward(self, token_ids):
        return self.projection(self.norm(self.embedding(token_ids)))


class NonRegistrationOrderModel(nn.Module):
    """Run layers in a different order from their attribute registration."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(5, 5, bias=False)
        self.second = nn.Linear(5, 5, bias=False)
        self.third = nn.Linear(5, 5, bias=False)

    def forward(self, inputs):
        return self.second(self.third(self.first(inputs)))


def test_fsdp_shards_padding_and_optimizer_state():
    world_size = 2
    mp.spawn(_test_padding_and_optimizer_state, args=(world_size,), nprocs=world_size, join=True)


def _test_padding_and_optimizer_state(rank: int, world_size: int) -> None:
    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    model = OddSizedModel()
    expected_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    fsdp_model = FullyShardedDataParallel(model)
    wrapped_model = fsdp_model.module
    assert isinstance(wrapped_model, OddSizedModel)

    assert wrapped_model.embedding.weight.numel() == 18  # ceil(7 * 5 / 2)
    assert wrapped_model.projection.weight.numel() == 8  # ceil(3 * 5 / 2)
    assert wrapped_model.norm.weight.shape == (5,)

    gathered = fsdp_model.gather_full_parameters()
    for name, expected in expected_parameters.items():
        torch.testing.assert_close(gathered[name], expected)

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-2, foreach=False)
    input_ids = torch.tensor([[rank, rank + 1], [rank + 2, rank + 3]])
    optimizer.zero_grad(set_to_none=True)
    loss = fsdp_model(input_ids).square().mean()
    loss.backward()
    fsdp_model.finish_gradient_synchronization()

    assert wrapped_model.embedding.weight.grad is not None
    assert wrapped_model.embedding.weight.grad.shape == wrapped_model.embedding.weight.shape
    assert wrapped_model.projection.weight.grad is not None
    assert wrapped_model.projection.weight.grad.shape == wrapped_model.projection.weight.shape
    assert wrapped_model.norm.weight.grad is not None

    optimizer.step()
    assert set(optimizer.state) == {
        wrapped_model.embedding.weight,
        wrapped_model.norm.weight,
        wrapped_model.projection.weight,
    }
    for parameter, state in optimizer.state.items():
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape

    for name, full_parameter in fsdp_model.gather_full_parameters().items():
        rank_zero_parameter = full_parameter.clone()
        dist.broadcast(rank_zero_parameter, src=0)
        torch.testing.assert_close(full_parameter, rank_zero_parameter)

    _cleanup_process_group()


def test_fsdp_prefetch_uses_observed_execution_order():
    world_size = 2
    mp.spawn(_test_non_registration_execution_order, args=(world_size,), nprocs=world_size, join=True)


def _test_non_registration_execution_order(rank: int, world_size: int) -> None:
    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(7)
    base_model = NonRegistrationOrderModel()
    reference_model = deepcopy(base_model)
    fsdp_model = FullyShardedDataParallel(base_model)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-2)
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)
    inputs = torch.randn(3, 5)

    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_model(inputs).square().mean().backward()
        reference_optimizer.step()

        fsdp_optimizer.zero_grad(set_to_none=True)
        fsdp_model(inputs).square().mean().backward()
        fsdp_model.finish_gradient_synchronization()
        fsdp_optimizer.step()

        full_parameters = fsdp_model.gather_full_parameters()
        for name, reference_parameter in reference_model.named_parameters():
            torch.testing.assert_close(full_parameters[name], reference_parameter)

    _cleanup_process_group()
