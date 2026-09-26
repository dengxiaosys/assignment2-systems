import torch

from cs336_systems.rmsnorm_fusion import (
    RMSNormResult,
    RMSNormVariant,
    RMSNormFusionWorkload,
    compare_rmsnorm_results,
)
from cs336_systems.saved_tensor_profiler import SavedTensorProfile, capture_saved_tensors


def _run_measured(
    input_data: torch.Tensor,
    *,
    variant: RMSNormVariant,
    compile_backend: str | None,
) -> tuple[RMSNormResult, SavedTensorProfile]:
    workload = RMSNormFusionWorkload(
        hidden_size=input_data.shape[-1],
        eps=1e-5,
        variant=variant,
        compile_backend=compile_backend,
    )
    workload.warm_up(input_data)
    x = input_data.detach().clone().requires_grad_(True)
    return capture_saved_tensors(
        lambda: workload.run(x),
        tensor_roles={
            "input": (x,),
            "parameter": workload.parameter_tensors,
        },
    )


def test_eager_rmsnorm_saved_tensor_accounting() -> None:
    input_data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    _, profile = _run_measured(input_data, variant="eager", compile_backend=None)

    assert profile.metrics().tensor_count == 6
    assert len(profile.loaded_events) == 6
    assert profile.metrics().logical_bytes == 352
    assert profile.metrics().unique_storage_bytes == 232
    assert sum(event.role != "parameter" and event.shape == tuple(input_data.shape) for event in profile.saved_events) == 3
    assert sorted(event.save_index for event in profile.loaded_events) == [event.save_index for event in profile.saved_events]


def test_compiled_rmsnorm_matches_eager() -> None:
    torch.manual_seed(0)
    input_data = torch.randn(2, 3, 4)
    eager_result, eager_profile = _run_measured(
        input_data,
        variant="eager",
        compile_backend=None,
    )
    compiled_result, compiled_profile = _run_measured(
        input_data,
        variant="compiled",
        compile_backend="aot_eager",
    )

    comparison = compare_rmsnorm_results(eager_result, compiled_result)

    assert comparison.output.allclose
    assert comparison.input_grad.allclose
    assert comparison.weight_grad.allclose
    assert compiled_profile.metrics().tensor_count < eager_profile.metrics().tensor_count
