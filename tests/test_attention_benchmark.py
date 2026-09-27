from cs336_systems.attention_benchmark import (
    AttentionBenchmarkConfig,
    attention_size_bytes,
    benchmark_attention_case,
)


def test_attention_size_bytes() -> None:
    config = AttentionBenchmarkConfig(
        batch_size=2,
        sequence_length=8,
        d_model=4,
        warmup_steps=0,
        measurement_steps=1,
    )

    sizes = attention_size_bytes(config)

    assert sizes["one_qkv_or_output_tensor"] == 2 * 8 * 4 * 4
    assert sizes["qkv_tensors"] == 3 * 2 * 8 * 4 * 4
    assert sizes["attention_score_tensor"] == 2 * 8 * 8 * 4


def test_benchmark_attention_case_reports_timings_and_saved_tensors() -> None:
    config = AttentionBenchmarkConfig(
        batch_size=2,
        sequence_length=8,
        d_model=4,
        warmup_steps=1,
        measurement_steps=2,
    )

    result = benchmark_attention_case(config)

    assert result["status"] == "ok"
    assert result["timings"]["forward"]["mean_ms"] > 0
    assert result["timings"]["backward"]["mean_ms"] > 0
    assert result["saved_tensors_after_forward"]["reference_count"] > 0
    assert result["saved_tensors_after_forward"]["unique_storage_bytes"] >= result["theoretical_bytes"]["attention_score_tensor"]
