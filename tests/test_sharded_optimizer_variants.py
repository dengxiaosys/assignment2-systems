from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.sharded_optimizer import ShardedOptimizer

from .common import _cleanup_process_group, _setup_process_group


def test_sharded_optimizer_supports_parameter_groups_and_partitions_state():
    world_size = 2
    mp.spawn(_test_parameter_groups_and_state_partition, args=(world_size,), nprocs=world_size, join=True)


def _test_parameter_groups_and_state_partition(rank: int, world_size: int) -> None:
    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    baseline_parameters = nn.ParameterList(
        [
            nn.Parameter(torch.randn(17)),
            nn.Parameter(torch.randn(11)),
            nn.Parameter(torch.randn(7)),
            nn.Parameter(torch.randn(3)),
        ]
    )
    sharded_parameters = nn.ParameterList([nn.Parameter(parameter.detach().clone()) for parameter in baseline_parameters])

    baseline_optimizer = torch.optim.AdamW(
        [{"params": baseline_parameters[:2], "lr": 0.05}],
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    sharded_optimizer = ShardedOptimizer(
        [{"params": sharded_parameters[:2], "lr": 0.05}],
        torch.optim.AdamW,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    baseline_optimizer.add_param_group({"params": baseline_parameters[2:], "lr": 0.01})
    sharded_optimizer.add_param_group({"params": sharded_parameters[2:], "lr": 0.01})
    baseline_scheduler = torch.optim.lr_scheduler.ExponentialLR(baseline_optimizer, gamma=0.5)
    sharded_scheduler = torch.optim.lr_scheduler.ExponentialLR(sharded_optimizer, gamma=0.5)

    gathered_owners: list[tuple[int, ...] | None] = [None] * world_size
    dist.all_gather_object(gathered_owners, sharded_optimizer.parameter_owners)
    assert all(owners == gathered_owners[0] for owners in gathered_owners)

    for step in range(3):
        for index, (baseline_parameter, sharded_parameter) in enumerate(zip(baseline_parameters, sharded_parameters, strict=True)):
            gradient = torch.full_like(baseline_parameter, (step + 1) * (index + 1) / 10)
            baseline_parameter.grad = gradient.clone()
            sharded_parameter.grad = gradient.clone()

        baseline_optimizer.step()
        sharded_optimizer.step()
        baseline_scheduler.step()
        sharded_scheduler.step()
        for baseline_parameter, sharded_parameter in zip(baseline_parameters, sharded_parameters, strict=True):
            torch.testing.assert_close(baseline_parameter, sharded_parameter)

    local_optimizer = sharded_optimizer.local_optimizer
    assert local_optimizer is not None
    locally_owned = {parameter for group in local_optimizer.param_groups for parameter in group["params"]}
    expected_locally_owned = {parameter for parameter, owner in zip(sharded_parameters, sharded_optimizer.parameter_owners, strict=True) if owner == rank}
    assert locally_owned == expected_locally_owned
    assert set(local_optimizer.state) == expected_locally_owned

    local_exp_avg_elements = sum(state["exp_avg"].numel() for state in local_optimizer.state.values())
    total_exp_avg_elements = torch.tensor(local_exp_avg_elements)
    dist.all_reduce(total_exp_avg_elements, op=dist.ReduceOp.SUM)
    assert total_exp_avg_elements.item() == sum(parameter.numel() for parameter in sharded_parameters)

    sharded_optimizer.zero_grad()
    assert all(parameter.grad is None for parameter in sharded_parameters)
    _cleanup_process_group()
