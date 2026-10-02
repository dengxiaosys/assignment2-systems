"""Distributed data-parallel strategies used by the Assignment 2 experiments."""

from __future__ import annotations

from typing import Any, Literal

import torch.distributed as dist
from torch import Tensor, nn

type DDPVariant = Literal["naive"]


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


def wrap_ddp(module: nn.Module, variant: DDPVariant) -> _DistributedDataParallelBase:
    """Construct the experiment's naive DDP strategy."""

    implementations = {"naive": NaiveDistributedDataParallel}
    return implementations[variant](module)
