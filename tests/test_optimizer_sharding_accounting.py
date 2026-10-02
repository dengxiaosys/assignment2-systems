import pytest

from cs336_systems.benchmark import ModelConfig
from cs336_systems.optimizer_sharding_accounting import (
    build_static_accounting,
    summarize_samples,
)


def test_summarize_optimizer_sharding_samples():
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["std_ms"] == pytest.approx(1.11803398875)
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["max_ms"] == pytest.approx(4.0)


def test_static_accounting_partitions_one_copy_of_optimizer_state():
    accounting = build_static_accounting(
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

    parameter_bytes = accounting["parameter_bytes_per_rank"]
    sharded_ranks = accounting["sharded_ranks"]
    assert accounting["parameter_count"] == 18_528
    assert parameter_bytes == 74_112
    assert sum(rank["owned_parameter_bytes"] for rank in sharded_ranks) == parameter_bytes
    assert sum(rank["adamw_moment_bytes"] for rank in sharded_ranks) == accounting["baseline_per_rank"]["adamw_moment_bytes"]
    assert all(rank["persistent_parameter_gradient_moment_bytes"] == 2 * parameter_bytes + rank["adamw_moment_bytes"] for rank in sharded_ranks)


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_summarize_optimizer_sharding_samples_rejects_invalid_input(samples):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_samples(samples)
