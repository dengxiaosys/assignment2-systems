import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.benchmark import ModelConfig
from cs336_systems.fsdp import FullyShardedDataParallel
from cs336_systems.fsdp_accounting import FSDPCommunicationRecorder, build_fsdp_static_accounting

from .common import _cleanup_process_group, _setup_process_group


def test_fsdp_static_accounting_shards_weights_and_replicates_norms():
    accounting = build_fsdp_static_accounting(
        model_config=ModelConfig(
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_heads=4,
        ),
        vocab_size=128,
        context_length=16,
        world_size=2,
    )

    assert accounting["parameter_count"] == 18_528
    assert accounting["parameter_tensors"] == 12
    assert accounting["shardable_parameter_tensors"] == 9
    assert accounting["replicated_parameter_tensors"] == 3
    assert accounting["full_parameter_bytes"] == 74_112
    assert accounting["replicated_parameter_bytes"] == 384
    assert accounting["fsdp_local_parameter_bytes_per_rank"] == 37_248
    assert accounting["fsdp_persistent_parameter_gradient_moment_bytes_per_rank"] == 148_992
    assert accounting["persistent_savings_vs_replicated_ddp_percent"] == pytest.approx(49.740933)
    assert accounting["peak_savings_vs_optimizer_sharding_rank_max_percent"] > 30


def test_fsdp_static_accounting_rejects_invalid_world_size():
    with pytest.raises(ValueError, match="must be positive"):
        build_fsdp_static_accounting(
            model_config=ModelConfig(
                d_model=32,
                d_ff=64,
                num_layers=1,
                num_heads=4,
            ),
            vocab_size=128,
            context_length=16,
            world_size=0,
        )


def test_fsdp_communication_recorder_observes_without_core_statistics():
    world_size = 2
    mp.spawn(
        _test_fsdp_communication_recorder,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_communication_recorder(rank: int, world_size: int) -> None:
    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    recorder = FSDPCommunicationRecorder()
    model = FullyShardedDataParallel(
        nn.Sequential(
            nn.Linear(5, 7, bias=False),
            nn.Linear(7, 3, bias=False),
        ),
        observer=recorder,
    )

    model(torch.randn(2, 5)).square().mean().backward()
    model.finish_gradient_synchronization()

    snapshot = recorder.snapshot(reset=True)
    records = snapshot["all_gather_records"]
    assert snapshot["all_gather_calls"] == 4
    assert [record["phase"] for record in records].count("forward") == 2
    assert [record["phase"] for record in records].count("backward") == 2
    assert all(record["communicated_dtype"] == "torch.float32" for record in records)
    assert recorder.snapshot()["all_gather_calls"] == 0
    _cleanup_process_group()
