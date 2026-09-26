import pytest
import torch
from torch import nn

from cs336_systems.saved_tensor_profiler import capture_saved_tensors


def test_capture_saved_tensors_records_roles_and_storage_metrics() -> None:
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    weight = nn.Parameter(torch.tensor([4.0, 5.0]))

    def operation() -> torch.Tensor:
        (x * weight).sum().backward()
        assert x.grad is not None
        return x.grad.detach().clone()

    gradient, profile = capture_saved_tensors(
        operation,
        tensor_roles={
            "input": (x,),
            "parameter": (weight,),
        },
        source_resolver=lambda _tensor: "test_source",
    )

    assert torch.equal(gradient, weight)
    assert profile.metrics().tensor_count == 2
    assert profile.metrics().logical_bytes == 16
    assert profile.metrics().unique_storage_bytes == 16
    assert profile.metrics(excluding_roles=("parameter",)).logical_bytes == 8
    assert {event.role for event in profile.saved_events} == {"input", "parameter"}
    assert {event.source for event in profile.events} == {"test_source"}
    assert all(event.grad_fn is None for event in profile.loaded_events)


def test_capture_saved_tensors_rejects_conflicting_storage_roles() -> None:
    x = torch.ones(2, requires_grad=True)

    with pytest.raises(ValueError, match="conflicting roles"):
        capture_saved_tensors(
            lambda: x.sum(),
            tensor_roles={
                "input": (x,),
                "parameter": (x,),
            },
        )
