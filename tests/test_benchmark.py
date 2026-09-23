import pytest
import torch

from cs336_systems.benchmark import ModelConfig, run_benchmark


@pytest.mark.parametrize(
    ("mode", "expected_phases"),
    [
        ("forward", {"forward", "total"}),
        ("forward-backward", {"forward", "loss", "backward", "total"}),
        ("full", {"forward", "loss", "backward", "optimizer", "total"}),
    ],
)
def test_run_benchmark_reports_expected_phases(mode, expected_phases):
    result = run_benchmark(
        model_size="test",
        model_config=ModelConfig(d_model=16, d_ff=32, num_layers=1, num_heads=2),
        mode=mode,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dtype_name="float32",
        vocab_size=32,
        batch_size=2,
        context_length=8,
        rope_theta=10_000.0,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=1,
        measurement_steps=2,
        seed=0,
    )

    assert result.parameter_count > 0
    assert set(result.phases) == expected_phases
    for phase in result.phases.values():
        assert len(phase.samples_ms) == 2
        assert phase.mean_ms >= 0
        assert phase.std_ms >= 0
