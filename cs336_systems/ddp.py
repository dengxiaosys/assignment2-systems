"""Distributed data-parallel strategies used by the Assignment 2 experiments."""

from __future__ import annotations

from typing import Any, Literal

import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.hooks import RemovableHandle

type DDPVariant = Literal["naive", "flat", "overlap"]


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


class OverlappedDistributedDataParallel(_DistributedDataParallelBase):
    """Launch an asynchronous all-reduce when each parameter gradient is ready."""

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._pending_reductions: list[tuple[dist.Work, Tensor]] = []
        self._pending_gradient_ids: set[int] = set()
        self._hook_handles: list[RemovableHandle] = [
            parameter.register_post_accumulate_grad_hook(self._all_reduce_gradient) for parameter in self._parameters_to_sync if parameter.requires_grad
        ]

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self._pending_reductions:
            raise RuntimeError("finish_gradient_synchronization() must be called before the next forward pass")
        return super().forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        try:
            for work, gradient in self._pending_reductions:
                work.wait()
                gradient.div_(self._world_size)
        finally:
            self._pending_reductions.clear()
            self._pending_gradient_ids.clear()

    def _all_reduce_gradient(self, parameter: Tensor) -> None:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("post-accumulate hook ran without an accumulated gradient")
        if gradient.is_sparse:
            raise RuntimeError("OverlappedDistributedDataParallel only supports dense gradients")
        gradient_id = id(gradient)
        if gradient_id in self._pending_gradient_ids:
            raise RuntimeError("a gradient was accumulated again before its previous all-reduce finished")

        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_reductions.append((work, gradient))
        self._pending_gradient_ids.add(gradient_id)


def wrap_ddp(module: nn.Module, variant: DDPVariant) -> _DistributedDataParallelBase:
    """Construct one of the experiment's DDP strategies."""

    implementations = {
        "naive": NaiveDistributedDataParallel,
        "flat": FlatDistributedDataParallel,
        "overlap": OverlappedDistributedDataParallel,
    }
    return implementations[variant](module)
