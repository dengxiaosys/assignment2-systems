from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from cs336_systems.ddp import DDPVariant, wrap_ddp

from .common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

VARIANTS: tuple[DDPVariant, ...] = ("naive", "flat")
MODEL_TYPES = (ToyModel, ToyModelWithTiedWeights)


def test_naive_and_flat_ddp_match_the_global_batch_baseline():
    world_size = 2
    mp.spawn(_test_naive_and_flat_ddp, args=(world_size,), nprocs=world_size, join=True)


def _test_naive_and_flat_ddp(rank: int, world_size: int) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt").to(device)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt").to(device)
    local_batch_size = all_x.shape[0] // world_size
    loss_fn = nn.MSELoss()

    for variant in VARIANTS:
        for model_type in MODEL_TYPES:
            torch.manual_seed(rank)
            baseline = model_type().to(device)
            ddp_model = wrap_ddp(deepcopy(baseline), variant)
            if isinstance(ddp_model.module, ToyModelWithTiedWeights):
                assert ddp_model.module.fc4.weight is ddp_model.module.fc2.weight

            baseline_optimizer = optim.SGD(baseline.parameters(), lr=0.1)
            ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
            for _ in range(2):
                baseline_optimizer.zero_grad()
                baseline_loss = loss_fn(baseline(all_x), all_y)
                baseline_loss.backward()
                baseline_optimizer.step()

                offset = rank * local_batch_size
                ddp_optimizer.zero_grad()
                local_output = ddp_model(all_x[offset : offset + local_batch_size])
                local_loss = loss_fn(local_output, all_y[offset : offset + local_batch_size])
                local_loss.backward()
                ddp_model.finish_gradient_synchronization()
                ddp_optimizer.step()

                validate_ddp_net_equivalence(ddp_model)
                if rank == 0:
                    for expected, actual in zip(baseline.parameters(), ddp_model.parameters(), strict=True):
                        assert torch.allclose(expected, actual), f"{variant=} {model_type.__name__=}"

            dist.barrier()

    _cleanup_process_group()
