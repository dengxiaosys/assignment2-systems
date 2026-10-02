"""Optimizer-state sharding for replicated data-parallel models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

type Params = Iterable[Tensor] | Iterable[dict[str, Any]]
type Closure = Callable[[], float]
type OptimizerStep = Callable[..., float | None]


class ShardedOptimizer(Optimizer):
    """Keep optimizer state only for locally owned replicated parameters.

    Every rank still holds the full parameter set. Parameters are assigned to
    one owner rank, the local optimizer updates only owned parameters, and each
    updated parameter is then broadcast from its owner to every other rank.
    """

    def __init__(
        self,
        params: Params,
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("a distributed process group must be initialized before constructing ShardedOptimizer")

        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        # Every rank derives the same ownership and ordering independently.
        # Together they define the source and sequence of parameter broadcasts.
        self._parameter_to_owner_rank: dict[Tensor, int] = {}
        self._parameters_in_broadcast_order: list[Tensor] = []
        self._rank_to_owned_parameter_bytes = [0] * self._world_size

        self._local_optimizer: Optimizer | None = None

        # Example with three normalized full parameter groups:
        #   full 0: {"params": [embedding],      "lr": 1e-4}
        #   full 1: {"params": [layer1, layer2], "lr": 3e-4}
        #   full 2: {"params": [lm_head],        "lr": 1e-3}
        # Suppose rank 0 owns embedding/layer1 and rank 1 owns layer2/lm_head.
        # Then:
        #   rank 0 local groups: [[embedding], [layer1]], mapping: [0, 1]
        #   rank 1 local groups: [[layer2], [lm_head]],   mapping: [1, 2]
        # On rank 1, local group 0 therefore reads options from full group 1,
        # rather than incorrectly reading them from full group 0.
        self._local_group_to_full_group_index: list[int] = []

        # Optimizer.__init__ dispatches to this class's add_param_group().
        self._is_building_initial_param_groups = True
        super().__init__(params, defaults=dict(kwargs))
        self._is_building_initial_param_groups = False
        self._build_local_optimizer()

    # Public optimizer API

    @property
    def local_optimizer(self) -> Optimizer | None:
        """Return the optimizer that owns this rank's state shard."""

        return self._local_optimizer

    @property
    def parameter_owners(self) -> tuple[int, ...]:
        """Return owner ranks in the stable parameter broadcast order."""

        return tuple(self._parameter_to_owner_rank[parameter] for parameter in self._parameters_in_broadcast_order)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Register a full parameter group and add its owned subset locally.

        PyTorch represents one optimizer configuration as a dictionary:

            {
                "params": [parameter_0, parameter_1, ...],
                "lr": 1e-3,
                "weight_decay": 0.01,
                # Other optimizer-specific options, such as AdamW betas.
            }

        ``Optimizer.param_groups`` is a list of these dictionaries, allowing
        different model parts to use different hyperparameters. This wrapper
        keeps the complete group in ``self.param_groups`` so inherited methods
        such as ``zero_grad()`` see every replicated parameter. The wrapped
        local optimizer receives a copy containing only this rank's parameters.
        """

        super().add_param_group(dict(param_group))
        # The parent method validates, normalizes, and appends this group.
        normalized_full_group = self.param_groups[-1]
        self._assign_owner_ranks(normalized_full_group["params"])

        if not self._is_building_initial_param_groups:
            self._add_local_param_group(normalized_full_group)

    def step(self, closure: Closure | None = None, **kwargs: Any) -> float | None:  # ty: ignore[invalid-method-override]
        """Update the local shard, then synchronize every replicated parameter."""

        if self._local_optimizer is None:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
        else:
            self._sync_local_group_options()
            optimizer_step = cast(OptimizerStep, self._local_optimizer.step)
            if closure is None:
                loss = optimizer_step(**kwargs)
            else:
                loss = optimizer_step(closure=closure, **kwargs)

        for parameter in self._parameters_in_broadcast_order:
            dist.broadcast(parameter.detach(), src=self._parameter_to_owner_rank[parameter])
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer-state shard."""

        if self._local_optimizer is None:
            return {"state": {}, "param_groups": []}
        self._sync_local_group_options()
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a rank-local optimizer-state shard."""

        if self._local_optimizer is None:
            if state_dict["state"] or state_dict["param_groups"]:
                raise ValueError("cannot load non-empty optimizer state on a rank with no owned parameters")
            return
        self._local_optimizer.load_state_dict(state_dict)
        self._adopt_loaded_group_options()
        self.state = self._local_optimizer.state

    # Parameter ownership and local optimizer construction

    def _assign_owner_ranks(self, parameters: list[Tensor]) -> None:
        for parameter in parameters:
            if parameter in self._parameter_to_owner_rank:
                continue
            owner_rank = min(range(self._world_size), key=lambda rank: (self._rank_to_owned_parameter_bytes[rank], rank))
            self._parameter_to_owner_rank[parameter] = owner_rank
            self._parameters_in_broadcast_order.append(parameter)
            self._rank_to_owned_parameter_bytes[owner_rank] += parameter.numel() * parameter.element_size()

    def _add_local_param_group(self, normalized_full_group: dict[str, Any]) -> None:
        """Add the current rank's part of a newly registered full group.

        For example, suppose full group 2 is
        ``{"params": [lm_head], "lr": 1e-3}`` and rank 1 owns ``lm_head``.
        Rank 1 adds ``{"params": [lm_head], "lr": 1e-3}`` to its local
        optimizer, while rank 0 adds nothing.
        """

        local_group = self._local_param_group(normalized_full_group)
        if local_group is None:
            return
        # Continuing the example above, rank 1's local groups become
        # [[layer2], [lm_head]]. Appending 2 makes its mapping [1, 2], so the
        # new [lm_head] local group reads options from full group 2.
        self._local_group_to_full_group_index.append(len(self.param_groups) - 1)
        if self._local_optimizer is None:
            self._local_optimizer = self._optimizer_cls([local_group], **self._optimizer_kwargs)
        else:
            self._local_optimizer.add_param_group(local_group)
        self._adopt_local_optimizer_state()

    def _local_param_group(self, normalized_full_group: dict[str, Any]) -> dict[str, Any] | None:
        """Project a normalized full group onto this rank's owned parameters.

        For example, full group 1 is
        ``{"params": [layer1, layer2], "lr": 3e-4}``. If rank 0 owns
        ``layer1`` and rank 1 owns ``layer2``, this returns
        ``{"params": [layer1], "lr": 3e-4}`` on rank 0 and
        ``{"params": [layer2], "lr": 3e-4}`` on rank 1.
        """

        owned_parameter_indices = [index for index, parameter in enumerate(normalized_full_group["params"]) if self._parameter_to_owner_rank[parameter] == self._rank]
        if not owned_parameter_indices:
            return None

        local_group = {name: value for name, value in normalized_full_group.items() if name not in {"params", "param_names"}}
        local_group["params"] = [normalized_full_group["params"][index] for index in owned_parameter_indices]
        if "param_names" in normalized_full_group:
            local_group["param_names"] = [normalized_full_group["param_names"][index] for index in owned_parameter_indices]
        return local_group

    def _adopt_local_optimizer_state(self) -> None:
        """Expose the wrapped local optimizer through the outer Optimizer API.

        For example, AdamW may add defaults such as ``betas`` and ``eps`` that
        were omitted from the original full groups. This method fills those
        defaults into every full group, then makes ``self.state`` reference the
        local optimizer's state dictionary. On rank 0 that dictionary might
        contain states for ``embedding`` and ``layer1`` only; it is not a copy
        of the full cross-rank optimizer state.
        """

        if self._local_optimizer is None:
            return
        self.defaults = self._local_optimizer.defaults  # 使用底层 optimizer 的完整默认超参数
        for group in self.param_groups:  # 遍历外层 wrapper 的完整参数组
            for name, value in self.defaults.items():  # 遍历 lr、betas、eps 等默认配置
                group.setdefault(name, value)  # 补充缺失项，不覆盖参数组的显式配置
        self.state = self._local_optimizer.state

    def _build_local_optimizer(self) -> None:
        local_groups = []
        for full_group_index, normalized_full_group in enumerate(self.param_groups):
            local_group = self._local_param_group(normalized_full_group)
            if local_group is None:
                continue
            self._local_group_to_full_group_index.append(full_group_index)
            local_groups.append(local_group)
        if not local_groups:
            return
        self._local_optimizer = self._optimizer_cls(local_groups, **self._optimizer_kwargs)
        self._adopt_local_optimizer_state()

    # Full/local optimizer state synchronization

    def _sync_local_group_options(self) -> None:
        if self._local_optimizer is None:
            return
        for full_group_index, local_group in zip(
            self._local_group_to_full_group_index,
            self._local_optimizer.param_groups,
            strict=True,
        ):
            normalized_full_group = self.param_groups[full_group_index]
            for name, value in normalized_full_group.items():
                if name not in {"params", "param_names"}:
                    local_group[name] = value

    def _adopt_loaded_group_options(self) -> None:
        if self._local_optimizer is None:
            return
        for full_group_index, local_group in zip(
            self._local_group_to_full_group_index,
            self._local_optimizer.param_groups,
            strict=True,
        ):
            normalized_full_group = self.param_groups[full_group_index]
            for name, value in local_group.items():
                if name not in {"params", "param_names"}:
                    normalized_full_group[name] = value
