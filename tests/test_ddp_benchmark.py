import math

import pytest

from cs336_systems.benchmark import ModelConfig
from cs336_systems.ddp_benchmark import DDPBenchmarkConfig, summarize_samples


def _config(*, global_batch_size: int = 4) -> DDPBenchmarkConfig:
    return DDPBenchmarkConfig(
        variant="naive",
        backend="gloo",
        world_size=2,
        model_size="custom",
        model_config=ModelConfig(d_model=64, d_ff=128, num_layers=2, num_heads=4),
        vocab_size=256,
        global_batch_size=global_batch_size,
        context_length=32,
        measurement_steps=10,
    )


def test_summarize_samples_reports_population_statistics_and_interpolated_p95():
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "count": 4,
        "mean_ms": 2.5,
        "std_ms": math.sqrt(1.25),
        "median_ms": 2.5,
        "p95_ms": 3.8499999999999996,
        "min_ms": 1.0,
        "max_ms": 4.0,
    }


@pytest.mark.parametrize("samples", [[], [-1.0], [math.inf], [math.nan]])
def test_summarize_samples_rejects_invalid_timings(samples):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_samples(samples)


def test_config_validates_global_and_local_batch_sizes():
    config = _config()
    config.validate()
    assert config.local_batch_size == 2
