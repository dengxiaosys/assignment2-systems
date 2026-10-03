from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.pipeline_parallel import PipelineParallel, TransformerPipelineStage, balanced_layer_partition


def _build_model() -> TransformerLM:
    return TransformerLM(
        vocab_size=64,
        context_length=8,
        d_model=16,
        num_layers=4,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
        device="cpu",
        dtype=torch.float32,
    )


def _language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def test_balanced_layer_partition():
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=0).start == 0
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=0).end == 3
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=1).start == 3
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=1).end == 5
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=2).start == 5
    assert balanced_layer_partition(num_layers=7, num_stages=3, stage_index=2).end == 7


@pytest.mark.parametrize(
    ("num_layers", "num_stages", "stage_index"),
    [
        (0, 1, 0),
        (2, 0, 0),
        (2, 3, 0),
        (2, 2, -1),
        (2, 2, 2),
    ],
)
def test_balanced_layer_partition_rejects_invalid_inputs(num_layers: int, num_stages: int, stage_index: int):
    with pytest.raises(ValueError):
        balanced_layer_partition(
            num_layers=num_layers,
            num_stages=num_stages,
            stage_index=stage_index,
        )


@pytest.mark.parametrize("world_size", [2, 3])
def test_pipeline_parallel_matches_non_parallel_training(world_size: int):
    server = dist.TCPStore(
        "127.0.0.1",
        0,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=60),
    )
    mp.spawn(
        _test_pipeline_parallel,
        args=(world_size, server.port),
        nprocs=world_size,
        join=True,
    )


def _test_pipeline_parallel(rank: int, world_size: int, port: int) -> None:
    store = dist.TCPStore(
        "127.0.0.1",
        port,
        is_master=False,
        timeout=timedelta(seconds=60),
    )
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        full_model = _build_model()
        reference_model = deepcopy(full_model) if rank == 0 else None
        stage = TransformerPipelineStage.from_model(
            full_model,
            stage_index=rank,
            num_stages=world_size,
        )
        del full_model

        pipeline = PipelineParallel(stage)
        pipeline_optimizer = torch.optim.SGD(stage.parameters(), lr=1e-2)
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-2) if reference_model is not None else None

        torch.manual_seed(123)
        input_ids = torch.randint(0, 64, (8, 8))
        targets = torch.randint(0, 64, (8, 8))

        for _ in range(2):
            expected_loss = None
            if reference_model is not None and reference_optimizer is not None:
                reference_optimizer.zero_grad(set_to_none=True)
                reference_logits = reference_model(input_ids)
                expected_loss = _language_model_loss(reference_logits, targets)
                expected_loss.backward()
                reference_optimizer.step()

            pipeline_optimizer.zero_grad(set_to_none=True)
            pipeline_loss = pipeline.forward_backward(
                input_ids,
                targets,
                loss_fn=_language_model_loss,
                num_microbatches=4,
            )
            pipeline_optimizer.step()

            loss_for_rank_zero = pipeline_loss if pipeline_loss is not None else torch.zeros((), dtype=torch.float32)
            dist.broadcast(loss_for_rank_zero, src=world_size - 1)
            if expected_loss is not None:
                torch.testing.assert_close(loss_for_rank_zero, expected_loss.detach())

            local_state = {name: tensor.detach().clone() for name, tensor in stage.state_dict().items()}
            gathered_states: list[dict[str, torch.Tensor] | None] | None = [None] * world_size if rank == 0 else None
            dist.gather_object(local_state, gathered_states, dst=0)
            if reference_model is not None and gathered_states is not None:
                pipeline_state = {name: tensor for stage_state in gathered_states if stage_state is not None for name, tensor in stage_state.items()}
                assert pipeline_state.keys() == reference_model.state_dict().keys()
                for name, expected in reference_model.state_dict().items():
                    torch.testing.assert_close(pipeline_state[name], expected)
    finally:
        dist.destroy_process_group()
