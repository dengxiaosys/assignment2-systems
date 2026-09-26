"""Reusable instrumentation for tensors retained by reverse-mode autograd."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

type StorageKey = tuple[str, int]


@dataclass(frozen=True)
class SavedTensorEvent:
    """One pack or unpack event associated with an autograd saved tensor."""

    phase: str
    save_index: int
    storage_index: int
    role: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool
    grad_fn: str | None
    source: str | None
    tensor_nbytes: int
    storage_nbytes: int


@dataclass(frozen=True)
class SavedTensorMetrics:
    """Aggregate save metrics after optional role filtering."""

    tensor_count: int
    logical_bytes: int
    unique_storage_bytes: int


@dataclass(frozen=True)
class SavedTensorProfile:
    """Immutable save/load trace returned by ``capture_saved_tensors``."""

    events: tuple[SavedTensorEvent, ...]

    @property
    def saved_events(self) -> tuple[SavedTensorEvent, ...]:
        return tuple(event for event in self.events if event.phase == "save")

    @property
    def loaded_events(self) -> tuple[SavedTensorEvent, ...]:
        return tuple(event for event in self.events if event.phase == "load")

    def metrics(self, *, excluding_roles: Collection[str] = ()) -> SavedTensorMetrics:
        excluded = frozenset(excluding_roles)
        saves = tuple(event for event in self.saved_events if event.role not in excluded)
        storage_nbytes = {event.storage_index: event.storage_nbytes for event in saves}
        return SavedTensorMetrics(
            tensor_count=len(saves),
            logical_bytes=sum(event.tensor_nbytes for event in saves),
            unique_storage_bytes=sum(storage_nbytes.values()),
        )

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "saved_tensor_count": metrics.tensor_count,
            "loaded_tensor_count": len(self.loaded_events),
            "logical_saved_bytes": metrics.logical_bytes,
            "unique_saved_storage_bytes": metrics.unique_storage_bytes,
            "save_sequence": [event.save_index for event in self.saved_events],
            "load_sequence": [event.save_index for event in self.loaded_events],
            "events": [asdict(event) for event in self.events],
        }


@dataclass(frozen=True)
class _PackedTensor:
    value: Tensor
    save_index: int
    source: str | None


def _storage_key(tensor: Tensor) -> StorageKey:
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _grad_fn_name(tensor: Tensor) -> str | None:
    return None if tensor.grad_fn is None else type(tensor.grad_fn).__name__


class _SavedTensorRecorder:
    def __init__(
        self,
        tensor_roles: Mapping[str, Iterable[Tensor]],
        source_resolver: Callable[[Tensor], str] | None,
    ):
        self._events: list[SavedTensorEvent] = []
        self._role_by_storage = self._build_role_map(tensor_roles)
        self._storage_indices: dict[StorageKey, int] = {}
        self._source_resolver = source_resolver
        self._next_save_index = 0

    @staticmethod
    def _build_role_map(tensor_roles: Mapping[str, Iterable[Tensor]]) -> dict[StorageKey, str]:
        role_by_storage: dict[StorageKey, str] = {}
        for role, tensors in tensor_roles.items():
            for tensor in tensors:
                storage_key = _storage_key(tensor)
                existing_role = role_by_storage.get(storage_key)
                if existing_role is not None and existing_role != role:
                    raise ValueError(f"one storage cannot have conflicting roles: {existing_role!r} and {role!r}")
                role_by_storage[storage_key] = role
        return role_by_storage

    def _storage_index(self, tensor: Tensor) -> int:
        storage_key = _storage_key(tensor)
        if storage_key not in self._storage_indices:
            self._storage_indices[storage_key] = len(self._storage_indices)
        return self._storage_indices[storage_key]

    def _record(
        self,
        phase: str,
        save_index: int,
        tensor: Tensor,
        source: str | None,
    ) -> None:
        self._events.append(
            SavedTensorEvent(
                phase=phase,
                save_index=save_index,
                storage_index=self._storage_index(tensor),
                role=self._role_by_storage.get(_storage_key(tensor), "intermediate"),
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                requires_grad=tensor.requires_grad,
                grad_fn=_grad_fn_name(tensor),
                source=source,
                tensor_nbytes=tensor.numel() * tensor.element_size(),
                storage_nbytes=tensor.untyped_storage().nbytes(),
            )
        )

    def pack(self, tensor: Tensor) -> _PackedTensor:
        save_index = self._next_save_index
        self._next_save_index += 1
        source = None if self._source_resolver is None else self._source_resolver(tensor)
        self._record("save", save_index, tensor, source)
        return _PackedTensor(value=tensor.detach(), save_index=save_index, source=source)

    def unpack(self, packed: _PackedTensor) -> Tensor:
        self._record("load", packed.save_index, packed.value, packed.source)
        return packed.value

    def profile(self) -> SavedTensorProfile:
        return SavedTensorProfile(events=tuple(self._events))


def capture_saved_tensors[T](
    operation: Callable[[], T],
    *,
    tensor_roles: Mapping[str, Iterable[Tensor]] | None = None,
    source_resolver: Callable[[Tensor], str] | None = None,
) -> tuple[T, SavedTensorProfile]:
    """Run ``operation`` and capture autograd saved-tensor save/load events.

    ``tensor_roles`` labels known storages, such as an input or parameters.
    Every unlabelled storage receives the role ``"intermediate"``. An optional
    ``source_resolver`` attributes each save event to a caller-defined source.
    """

    recorder = _SavedTensorRecorder(
        {} if tensor_roles is None else tensor_roles,
        source_resolver,
    )
    with torch.autograd.graph.saved_tensors_hooks(recorder.pack, recorder.unpack):
        result = operation()
    return result, recorder.profile()
