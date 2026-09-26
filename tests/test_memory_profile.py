from scripts.profile_memory import MemorySample, _summarize_samples


def test_summarize_cpu_memory_samples() -> None:
    samples = [
        MemorySample(0.0, "baseline", 100, None, None),
        MemorySample(1.0, "forward", 150, None, None),
        MemorySample(2.0, "forward", 175, None, None),
        MemorySample(3.0, "backward", 160, None, None),
    ]

    summary = _summarize_samples(samples)

    assert summary["peak_rss_bytes"] == 175
    assert summary["peak_cuda_allocated_bytes"] is None
    assert summary["stage_peaks"]["forward"]["rss_bytes"] == 175
    assert summary["stage_peaks"]["backward"]["rss_bytes"] == 160


def test_summarize_cuda_memory_samples() -> None:
    samples = [
        MemorySample(0.0, "forward", 100, 20, 40),
        MemorySample(1.0, "forward", 110, 30, 60),
        MemorySample(2.0, "backward", 120, 25, 60),
    ]

    summary = _summarize_samples(samples)

    assert summary["peak_cuda_allocated_bytes"] == 30
    assert summary["peak_cuda_reserved_bytes"] == 60
    assert summary["stage_peaks"]["forward"]["cuda_allocated_bytes"] == 30
    assert summary["stage_peaks"]["backward"]["cuda_allocated_bytes"] == 25


def test_summarize_memory_samples_by_step() -> None:
    samples = [
        MemorySample(0.0, "forward", 100, None, None, step=1),
        MemorySample(1.0, "backward", 150, None, None, step=1),
        MemorySample(2.0, "forward", 120, None, None, step=2),
        MemorySample(3.0, "backward", 140, None, None, step=2),
    ]

    summary = _summarize_samples(samples)

    assert summary["step_stage_peaks"]["1"]["backward"]["rss_bytes"] == 150
    assert summary["step_stage_peaks"]["2"]["forward"]["rss_bytes"] == 120
