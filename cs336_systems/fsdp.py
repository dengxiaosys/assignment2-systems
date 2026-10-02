"""A teaching-oriented fully sharded data-parallel implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from cs336_basics.model import Embedding, Linear
from cs336_systems.fsdp_observer import AllGatherStage, FSDPAllGatherEvent, FSDPObserver, FSDPPhase

_SHARDABLE_MODULE_TYPES = (Linear, Embedding, nn.Linear, nn.Embedding)


@dataclass
class _PendingAllGather:
    work: dist.Work
    gathered_shards: list[Tensor]
    local_compute_shard: Tensor


@dataclass
class _PendingGradientReduction:
    work: dist.Work
    reduced_shard: Tensor
    parameter: nn.Parameter


class _ShardedWeight:
    """Runtime state for one sharded Linear or Embedding weight."""

    def __init__(
        self,
        *,
        module: nn.Module,
        parameter: nn.Parameter,
        module_name: str,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> None:
        self.module = module
        self.parameter = parameter
        self.module_name = module_name
        self.rank = rank
        self.world_size = world_size
        self.compute_dtype = compute_dtype
        self.full_shape = tuple(parameter.shape)
        self.full_numel = parameter.numel()
        self.shard_numel = (self.full_numel + world_size - 1) // world_size
        self.padded_numel = self.shard_numel * world_size

        flat_weight = parameter.detach().reshape(-1)
        padded_weight = flat_weight.new_zeros(self.padded_numel)
        padded_weight[: self.full_numel].copy_(flat_weight)
        start = rank * self.shard_numel
        parameter.data = padded_weight.narrow(0, start, self.shard_numel).clone()
        self.master_shard = parameter.data

        self.pending_all_gather: _PendingAllGather | None = None
        self.is_materialized = False

    @property
    def communicated_dtype(self) -> torch.dtype:
        return self.compute_dtype or self.master_shard.dtype

    def start_all_gather(self) -> _PendingAllGather | None:
        if self.pending_all_gather is not None or self.is_materialized:
            return None

        local_compute_shard = self.master_shard.to(self.communicated_dtype)
        gathered_shards = [torch.empty_like(local_compute_shard) for _ in range(self.world_size)]
        work = dist.all_gather(gathered_shards, local_compute_shard, async_op=True)
        pending_all_gather = _PendingAllGather(
            work=work,
            gathered_shards=gathered_shards,
            local_compute_shard=local_compute_shard,
        )
        self.pending_all_gather = pending_all_gather
        return pending_all_gather

    def materialize(self) -> None:
        """Wait for the pending all-gather and install the full compute weight."""

        if self.is_materialized:
            return
        if self.pending_all_gather is None:
            raise RuntimeError(f"all-gather has not started for {self.module_name}")

        self.pending_all_gather.work.wait()
        full_flat = torch.cat(self.pending_all_gather.gathered_shards)
        self.parameter.data = full_flat[: self.full_numel].reshape(self.full_shape)
        self.pending_all_gather = None
        self.is_materialized = True

    def reshard(self) -> None:
        self.parameter.data = self.master_shard
        self.is_materialized = False

    def padded_flat_gradient(self) -> Tensor:
        gradient = self.parameter.grad
        if gradient is None:
            raise RuntimeError(f"weight gradient is missing for {self.module_name}")
        if gradient.is_sparse:
            raise RuntimeError("FullyShardedDataParallel only supports dense gradients")

        flat_gradient = gradient.detach().reshape(-1).to(self.master_shard.dtype)
        padded_gradient = flat_gradient.new_zeros(self.padded_numel)
        padded_gradient[: self.full_numel].copy_(flat_gradient)
        return padded_gradient

    def gather_full_master_weight(self) -> Tensor:
        gathered_shards = [torch.empty_like(self.master_shard) for _ in range(self.world_size)]
        dist.all_gather(gathered_shards, self.master_shard)
        return torch.cat(gathered_shards)[: self.full_numel].reshape(self.full_shape)


class FullyShardedDataParallel(nn.Module):
    """Shard Linear/Embedding weights while replicating small parameters.

    The optimizer sees FP32 master-weight shards. Hooks materialize full weights
    just before layer compute, reshard them after use, and reduce-scatter full
    weight gradients back to the master-shard layout.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        observer: FSDPObserver | None = None,
    ):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("a distributed process group must be initialized before constructing FullyShardedDataParallel")
        if compute_dtype is not None and compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("compute_dtype must be float16, bfloat16, float32, or None")

        self.module = module
        self.compute_dtype = compute_dtype
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._hook_handles: list[RemovableHandle] = []
        self._pending_gradient_reductions: list[_PendingGradientReduction] = []
        self._pending_replicated_reductions: list[tuple[dist.Work, Tensor]] = []
        self._observer = observer

        self._broadcast_initial_state()
        self._sharded_weights = self._build_sharded_weights()
        self._parameter_to_sharded_weight = {state.parameter: state for state in self._sharded_weights}
        self._forward_order: tuple[int, ...] | None = None
        self._forward_position_by_index: dict[int, int] = {}
        self._current_forward_order: list[int] = []
        self._backward_order: tuple[int, ...] | None = None
        self._current_backward_order: list[int] = []
        self._register_layer_hooks()
        self._register_gradient_hooks()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if self._pending_gradient_reductions or self._pending_replicated_reductions:
            raise RuntimeError("finish_gradient_synchronization() must be called before the next forward pass")
        self._reshard_all_weights()
        self._current_forward_order.clear()
        self._current_backward_order.clear()
        self._prefetch_forward_position(0)
        self._prefetch_forward_position(1)
        output = self.module(*inputs, **kwargs)
        self._finish_forward_order()
        return output

    def finish_gradient_synchronization(self) -> None:
        """Wait for gradient collectives and expose optimizer-ready gradients."""

        try:
            self._finish_backward_order()
            for pending in self._pending_gradient_reductions:
                pending.work.wait()
                pending.reduced_shard.div_(self._world_size)
                pending.parameter.grad = pending.reduced_shard
            for work, gradient in self._pending_replicated_reductions:
                work.wait()
                gradient.div_(self._world_size)
        finally:
            self._pending_gradient_reductions.clear()
            self._pending_replicated_reductions.clear()
            self._reshard_all_weights()

    def gather_full_parameters(self) -> dict[str, Tensor]:
        """Collect full parameter values without changing optimizer-visible shards."""

        sharded_by_parameter = self._parameter_to_sharded_weight
        return {
            name: (sharded_by_parameter[parameter].gather_full_master_weight() if parameter in sharded_by_parameter else parameter.detach().clone())
            for name, parameter in self.module.named_parameters()
        }

    def _broadcast_initial_state(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.detach(), src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.detach(), src=0)

    def _build_sharded_weights(self) -> list[_ShardedWeight]:
        sharded_weights = []
        seen_parameter_ids: set[int] = set()
        for module_name, child_module in self.module.named_modules():
            if not isinstance(child_module, _SHARDABLE_MODULE_TYPES):
                continue
            parameter = child_module.weight
            if id(parameter) in seen_parameter_ids:
                raise RuntimeError("FullyShardedDataParallel does not support a weight shared by multiple shardable modules")
            seen_parameter_ids.add(id(parameter))
            if parameter.dtype != torch.float32:
                raise ValueError("FSDP master weights must be float32")
            sharded_weights.append(
                _ShardedWeight(
                    module=child_module,
                    parameter=parameter,
                    module_name=module_name,
                    rank=self._rank,
                    world_size=self._world_size,
                    compute_dtype=self.compute_dtype,
                )
            )
        return sharded_weights

    def _register_layer_hooks(self) -> None:
        for index, state in enumerate(self._sharded_weights):
            self._hook_handles.append(state.module.register_forward_pre_hook(self._make_forward_pre_hook(index)))
            self._hook_handles.append(state.module.register_forward_hook(self._make_forward_post_hook(index)))

    def _register_gradient_hooks(self) -> None:
        for state in self._sharded_weights:
            self._hook_handles.append(state.parameter.register_post_accumulate_grad_hook(self._make_sharded_gradient_hook(state)))

        sharded_parameter_ids = {id(state.parameter) for state in self._sharded_weights}
        for parameter in self.module.parameters():
            if parameter.requires_grad and id(parameter) not in sharded_parameter_ids:
                self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._all_reduce_replicated_gradient))

    def _make_forward_pre_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._record_forward_use(index)
            self._materialize(index, phase="forward")

        return hook

    def _make_forward_post_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise RuntimeError("FSDP shardable modules must return a Tensor")
            if output.requires_grad:
                output.register_hook(self._make_output_gradient_hook(index))
            self._sharded_weights[index].reshard()
            self._prefetch_forward_after(index)

        return hook

    def _make_output_gradient_hook(self, index: int):
        def hook(output_gradient: Tensor) -> Tensor:
            position = self._record_backward_use(index)
            self._materialize(index, phase="backward")
            self._prefetch_backward_position(position + 1)
            self._prefetch_backward_position(position + 2)
            return output_gradient

        return hook

    def _make_sharded_gradient_hook(self, state: _ShardedWeight):
        def hook(parameter: Tensor) -> None:
            padded_gradient = state.padded_flat_gradient()
            reduced_shard = torch.empty_like(state.master_shard)
            work = dist.reduce_scatter_tensor(
                reduced_shard,
                padded_gradient,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            parameter.grad = None
            state.reshard()
            self._pending_gradient_reductions.append(
                _PendingGradientReduction(
                    work=work,
                    reduced_shard=reduced_shard,
                    parameter=state.parameter,
                )
            )

        return hook

    def _all_reduce_replicated_gradient(self, parameter: Tensor) -> None:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("post-accumulate hook ran without a replicated gradient")
        if gradient.is_sparse:
            raise RuntimeError("FullyShardedDataParallel only supports dense gradients")
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_replicated_reductions.append((work, gradient))

    def _materialize(self, index: int, *, phase: FSDPPhase) -> None:
        state = self._sharded_weights[index]
        if state.is_materialized:
            return

        self._start_all_gather(index, phase=phase)
        pending = state.pending_all_gather
        if pending is None:
            raise RuntimeError(f"all-gather is unavailable for {state.module_name}")
        self._notify_all_gather("waiting", phase, state, pending)
        state.materialize()
        self._notify_all_gather("finished", phase, state, pending)

    def _record_forward_use(self, index: int) -> None:
        position = len(self._current_forward_order)
        if self._forward_order is not None and (position >= len(self._forward_order) or self._forward_order[position] != index):
            raise RuntimeError("FullyShardedDataParallel requires the same shardable-module execution order on every forward pass")
        self._current_forward_order.append(index)

    def _finish_forward_order(self) -> None:
        observed_order = tuple(self._current_forward_order)
        if self._forward_order is None:
            if len(observed_order) != len(self._sharded_weights) or len(set(observed_order)) != len(observed_order):
                raise RuntimeError("FullyShardedDataParallel requires every shardable module to run exactly once per forward pass")
            self._forward_order = observed_order
            self._forward_position_by_index = {index: position for position, index in enumerate(observed_order)}
        elif observed_order != self._forward_order:
            raise RuntimeError("FullyShardedDataParallel requires a static shardable-module execution order")

    def _record_backward_use(self, index: int) -> int:
        position = len(self._current_backward_order)
        if self._backward_order is not None and (position >= len(self._backward_order) or self._backward_order[position] != index):
            raise RuntimeError("FullyShardedDataParallel requires the same shardable-module execution order on every backward pass")
        self._current_backward_order.append(index)
        return position

    def _finish_backward_order(self) -> None:
        observed_order = tuple(self._current_backward_order)
        if self._backward_order is None:
            if len(observed_order) != len(self._sharded_weights) or len(set(observed_order)) != len(observed_order):
                raise RuntimeError("FullyShardedDataParallel requires every shardable module to run exactly once per backward pass")
            self._backward_order = observed_order
        elif observed_order != self._backward_order:
            raise RuntimeError("FullyShardedDataParallel requires a static shardable-module execution order")

    def _prefetch_forward_position(self, position: int) -> None:
        if self._forward_order is not None and 0 <= position < len(self._forward_order):
            self._start_all_gather(self._forward_order[position], phase="forward")

    def _prefetch_forward_after(self, index: int) -> None:
        if self._forward_order is None:
            return
        self._prefetch_forward_position(self._forward_position_by_index[index] + 2)

    def _prefetch_backward_position(self, position: int) -> None:
        if self._backward_order is not None and 0 <= position < len(self._backward_order):
            self._start_all_gather(self._backward_order[position], phase="backward")

    def _start_all_gather(self, index: int, *, phase: FSDPPhase) -> None:
        if not 0 <= index < len(self._sharded_weights):
            return
        state = self._sharded_weights[index]
        pending = state.start_all_gather()
        if pending is not None:
            self._notify_all_gather("launched", phase, state, pending)

    def _notify_all_gather(
        self,
        stage: AllGatherStage,
        phase: FSDPPhase,
        state: _ShardedWeight,
        pending: _PendingAllGather,
    ) -> None:
        if self._observer is None:
            return
        self._observer(
            FSDPAllGatherEvent(
                stage=stage,
                phase=phase,
                module_name=state.module_name,
                communicated_dtype=state.communicated_dtype,
                communicated_numel=state.padded_numel,
                device=state.master_shard.device,
                work=pending.work,
            )
        )

    def _reshard_all_weights(self) -> None:
        for state in self._sharded_weights:
            if state.is_materialized:
                state.reshard()
