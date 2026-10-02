"""Observation seam for FSDP collective lifecycle events."""

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.distributed as dist

FSDPPhase = Literal["forward", "backward"]
AllGatherStage = Literal["launched", "waiting", "finished"]


@dataclass(frozen=True)
class FSDPAllGatherEvent:
    """One lifecycle event for an asynchronous weight all-gather."""

    stage: AllGatherStage
    phase: FSDPPhase
    module_name: str
    communicated_dtype: torch.dtype
    communicated_numel: int
    device: torch.device
    work: dist.Work


class FSDPObserver(Protocol):
    """Receive optional diagnostics without coupling them to FSDP execution."""

    def __call__(self, event: FSDPAllGatherEvent) -> None: ...
