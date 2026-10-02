"""Distributed data-parallel strategies used by the Assignment 2 experiments."""

from __future__ import annotations

from typing import Any, Literal

import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

type DDPVariant = Literal["naive", "flat"]

class _DistributedDataParallelBase(nn.Module):
    """Own shared replication behavior behind the training-facing interface."""

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("a distributed process group must be initialized before wrapping the module")

        self.module = module
        self._world_size = dist.get_world_size()
        self._parameters_to_sync = tuple(module.parameters())
        self._broadcast_module_state()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        raise NotImplementedError

    def _broadcast_module_state(self) -> None:
        for parameter in self._parameters_to_sync:
            dist.broadcast(parameter.detach(), src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.detach(), src=0)

    def _dense_gradients(self) -> tuple[Tensor, ...]:
        gradients = []
        for parameter in self._parameters_to_sync:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError(f"{type(self).__name__} only supports dense gradients")
            gradients.append(parameter.grad)
        return tuple(gradients)


class NaiveDistributedDataParallel(_DistributedDataParallelBase):
    """Synchronously all-reduce each parameter gradient after backward."""

    def finish_gradient_synchronization(self) -> None:
        for gradient in self._dense_gradients():
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(self._world_size)


class FlatDistributedDataParallel(_DistributedDataParallelBase):
    """Synchronously all-reduce one flattened gradient tensor after backward."""

    def finish_gradient_synchronization(self) -> None:
        gradients = self._dense_gradients()
        if not gradients:
            return
        self._validate_flattenable(gradients)

        flat_gradient = _flatten_dense_tensors(gradients)
        dist.all_reduce(flat_gradient, op=dist.ReduceOp.SUM)
        flat_gradient.div_(self._world_size)
        for gradient, synchronized in zip(
            gradients,
            _unflatten_dense_tensors(flat_gradient, gradients),
            strict=True,
        ):
            gradient.copy_(synchronized)

    @staticmethod
    def _validate_flattenable(gradients: tuple[Tensor, ...]) -> None:
        first = gradients[0]
        if any(gradient.device != first.device for gradient in gradients):
            raise RuntimeError("flat gradient synchronization requires all gradients to share one device")
        if any(gradient.dtype != first.dtype for gradient in gradients):
            raise RuntimeError("flat gradient synchronization requires all gradients to share one dtype")


def wrap_ddp(module: nn.Module, variant: DDPVariant) -> _DistributedDataParallelBase:
    """Construct one of the experiment's DDP strategies."""

    implementations = {
        "naive": NaiveDistributedDataParallel,
        "flat": FlatDistributedDataParallel,
    }
    return implementations[variant](module)
