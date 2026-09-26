"""Pure-FP32 RMSNorm workload used to compare eager and compiled execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

type RMSNormVariant = Literal["eager", "compiled"]


class RMSNorm(nn.Module):
    """RMSNorm implementation matching the Assignment 2 handout."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, device: torch.device | str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x * rms
        return self.weight * normalized


@dataclass(frozen=True)
class RMSNormResult:
    output: Tensor
    input_grad: Tensor
    weight_grad: Tensor


@dataclass(frozen=True)
class TensorComparison:
    allclose: bool
    max_abs_diff: float
    relative_l2_error: float
    rtol: float
    atol: float


@dataclass(frozen=True)
class RMSNormComparison:
    output: TensorComparison
    input_grad: TensorComparison
    weight_grad: TensorComparison

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNormFusionWorkload:
    """Own eager/compiled RMSNorm execution without profiling concerns."""

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        variant: RMSNormVariant,
        compile_backend: str | None,
        device: torch.device | str = "cpu",
    ):
        if hidden_size <= 0:
            raise ValueError("hidden size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.variant = variant
        self._module = RMSNorm(hidden_size, eps=eps, device=device)
        self._forward: Callable[[Tensor], Tensor] = self._module
        if variant == "compiled":
            if compile_backend is None:
                self._forward = torch.compile(self._module, fullgraph=True)
            else:
                self._forward = torch.compile(self._module, fullgraph=True, backend=compile_backend)
        elif variant != "eager":
            raise ValueError(f"unsupported variant: {variant}")

    @property
    def parameter_tensors(self) -> tuple[nn.Parameter, ...]:
        return tuple(self._module.parameters())

    def warm_up(self, input_data: Tensor) -> None:
        """Materialize lazy compiled forward/backward graphs outside measurement."""

        if self.variant == "eager":
            return
        warmup_input = self._new_input(input_data)
        self._forward(warmup_input).sum().backward()
        self._module.zero_grad(set_to_none=True)

    def run(self, x: Tensor) -> RMSNormResult:
        """Run one forward/backward step and return detached result snapshots."""

        self._validate_input(x)
        if not x.requires_grad:
            raise ValueError("input must require gradients")
        if not x.is_leaf:
            raise ValueError("input must be a leaf tensor so its gradient is retained")
        self._module.zero_grad(set_to_none=True)
        x.grad = None

        output = self._forward(x)
        output.sum().backward()
        if x.grad is None or self._module.weight.grad is None:
            raise RuntimeError("backward did not produce the expected gradients")
        return RMSNormResult(
            output=output.detach().clone(),
            input_grad=x.grad.detach().clone(),
            weight_grad=self._module.weight.grad.detach().clone(),
        )

    def _new_input(self, input_data: Tensor) -> Tensor:
        self._validate_input(input_data)
        return input_data.detach().clone().requires_grad_(True)

    def _validate_input(self, tensor: Tensor) -> None:
        if tensor.dtype is not torch.float32:
            raise ValueError("this workload intentionally uses pure FP32")
        if tensor.device != self._module.weight.device:
            raise ValueError("input and RMSNorm parameters must be on the same device")
        if tensor.ndim == 0 or tensor.shape[-1] != self._module.weight.numel():
            raise ValueError("input's final dimension must equal the RMSNorm hidden size")


def compare_rmsnorm_results(
    eager: RMSNormResult,
    compiled: RMSNormResult,
    *,
    rtol: float = 1e-5,
    atol: float = 5e-4,
) -> RMSNormComparison:
    def compare(left: Tensor, right: Tensor) -> TensorComparison:
        difference = left.double() - right.double()
        reference_norm = float(left.double().norm())
        relative_l2_error = float(difference.norm()) / reference_norm if reference_norm else 0.0
        return TensorComparison(
            allclose=torch.allclose(left, right, rtol=rtol, atol=atol),
            max_abs_diff=float(difference.abs().max()),
            relative_l2_error=relative_l2_error,
            rtol=rtol,
            atol=atol,
        )

    return RMSNormComparison(
        output=compare(eager.output, compiled.output),
        input_grad=compare(eager.input_grad, compiled.input_grad),
        weight_grad=compare(eager.weight_grad, compiled.weight_grad),
    )
