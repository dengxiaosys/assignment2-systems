import pytest

from cs336_systems.benchmark import ModelConfig
from cs336_systems.pipeline_parallel_accounting import PipelineAccountingConfig, build_pipeline_static_accounting


def _config(
    *,
    world_size: int = 2,
    num_microbatches: int = 4,
) -> PipelineAccountingConfig:
    return PipelineAccountingConfig(
        backend="gloo",
        world_size=world_size,
        model_config=ModelConfig(
            d_model=16,
            d_ff=32,
            num_layers=4,
            num_heads=4,
        ),
        vocab_size=64,
        global_batch_size=8,
        context_length=8,
        num_microbatches=num_microbatches,
    )


def test_pipeline_static_accounting_preserves_total_parameters():
    accounting = build_pipeline_static_accounting(_config())

    assert [stage["layer_count"] for stage in accounting["stages"]] == [2, 2]
    assert accounting["total_parameter_bytes"] == sum(stage["parameter_bytes"] for stage in accounting["stages"])
    assert accounting["max_stage_parameter_bytes"] - accounting["min_stage_parameter_bytes"] == 16 * 4
    assert accounting["forward_activation_bytes_per_boundary_per_batch"] == 8 * 8 * 16 * 4
    assert accounting["backward_activation_gradient_bytes_per_boundary_per_batch"] == 8 * 8 * 16 * 4
    assert accounting["bidirectional_boundary_bytes_per_batch"] == 2 * 8 * 8 * 16 * 4
    assert accounting["ideal_fill_drain_efficiency"] == pytest.approx(4 / 5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world_size", 1, "world_size"),
        ("world_size", 5, "world_size"),
        ("num_microbatches", 0, "num_microbatches"),
        ("num_microbatches", 3, "divisible"),
    ],
)
def test_pipeline_accounting_rejects_invalid_config(field: str, value: int, message: str):
    config = _config(world_size=value) if field == "world_size" else _config(num_microbatches=value)
    with pytest.raises(ValueError, match=message):
        build_pipeline_static_accounting(config)
