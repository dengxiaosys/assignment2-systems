"""A teaching-oriented GPipe-style pipeline-parallel implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from cs336_basics.model import Embedding, Linear, RMSNorm, TransformerBlock, TransformerLM

type LossFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class LayerPartition:
    """A contiguous half-open Transformer-block range owned by one stage."""

    stage_index: int
    num_stages: int
    start: int  # Inclusive global Transformer-block index.
    end: int  # Exclusive global Transformer-block index.

    @property
    def layer_count(self) -> int:
        return self.end - self.start


def balanced_layer_partition(
    *,
    num_layers: int,
    num_stages: int,
    stage_index: int,
) -> LayerPartition:
    """Assign contiguous layers while differing by at most one layer."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_stages <= 0 or num_stages > num_layers:
        raise ValueError("num_stages must be positive and no greater than num_layers")
    if not 0 <= stage_index < num_stages:
        raise ValueError("stage_index must be in [0, num_stages)")

    base_layers, extra_layers = divmod(num_layers, num_stages)
    start = stage_index * base_layers + min(stage_index, extra_layers)
    layer_count = base_layers + int(stage_index < extra_layers)
    return LayerPartition(
        stage_index=stage_index,
        num_stages=num_stages,
        start=start,
        end=start + layer_count,
    )


class TransformerPipelineStage(nn.Module):
    """Own one contiguous stage of a Transformer language model."""

    def __init__(
        self,
        *,
        partition: LayerPartition,
        d_model: int,
        layers: dict[str, nn.Module],
        token_embeddings: nn.Module | None,
        ln_final: nn.Module | None,
        lm_head: nn.Module | None,
    ) -> None:
        super().__init__()
        self.partition = partition
        self.d_model = d_model
        self.token_embeddings = token_embeddings
        self.layers = nn.ModuleDict(layers)
        self.ln_final = ln_final
        self.lm_head = lm_head

        if self.is_first_stage != (token_embeddings is not None):
            raise ValueError("only the first stage may own token_embeddings")
        if self.is_last_stage != (ln_final is not None and lm_head is not None):
            raise ValueError("only the last stage must own ln_final and lm_head")

    @property
    def stage_index(self) -> int:
        return self.partition.stage_index

    @property
    def num_stages(self) -> int:
        return self.partition.num_stages

    @property
    def is_first_stage(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage_index == self.num_stages - 1

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def activation_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_model(
        cls,
        model: TransformerLM,
        *,
        stage_index: int,
        num_stages: int,
    ) -> TransformerPipelineStage:
        """Transfer this stage's modules from an instantiated full model."""

        partition = balanced_layer_partition(
            num_layers=len(model.layers),
            num_stages=num_stages,
            stage_index=stage_index,
        )
        return cls(
            partition=partition,
            d_model=model.token_embeddings.weight.shape[1],
            layers={str(index): model.layers[index] for index in range(partition.start, partition.end)},
            token_embeddings=model.token_embeddings if stage_index == 0 else None,
            ln_final=model.ln_final if stage_index == num_stages - 1 else None,
            lm_head=model.lm_head if stage_index == num_stages - 1 else None,
        )

    @classmethod
    def from_dimensions(
        cls,
        *,
        stage_index: int,
        num_stages: int,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerPipelineStage:
        """Construct only the modules owned by this rank."""

        partition = balanced_layer_partition(
            num_layers=num_layers,
            num_stages=num_stages,
            stage_index=stage_index,
        )
        return cls(
            partition=partition,
            d_model=d_model,
            layers={
                str(index): TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for index in range(partition.start, partition.end)
            },
            token_embeddings=(Embedding(vocab_size, d_model, device=device, dtype=dtype) if stage_index == 0 else None),
            ln_final=(RMSNorm(d_model, device=device, dtype=dtype) if stage_index == num_stages - 1 else None),
            lm_head=(Linear(d_model, vocab_size, device=device, dtype=dtype) if stage_index == num_stages - 1 else None),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden_states = self.token_embeddings(inputs) if self.token_embeddings is not None else inputs
        for layer in self.layers.values():
            hidden_states = layer(hidden_states)
        if self.ln_final is not None and self.lm_head is not None:
            hidden_states = self.lm_head(self.ln_final(hidden_states))
        return hidden_states


@dataclass
class _MicrobatchState:
    input_activation: Tensor | None
    output: Tensor
    loss: Tensor | None


class PipelineParallel:
    """Run synchronous GPipe fill-drain training over distributed stages.

    Every rank calls ``forward_backward`` with the same batch metadata. Rank 0
    consumes token IDs, the last rank consumes targets, and intermediate ranks
    communicate only activations and activation gradients.
    """

    def __init__(self, stage: TransformerPipelineStage) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("a distributed process group must be initialized before constructing PipelineParallel")
        if dist.get_world_size() != stage.num_stages:
            raise ValueError("process-group world size must equal the number of pipeline stages")
        if dist.get_rank() != stage.stage_index:
            raise ValueError("the local rank must match the pipeline stage index")

        self.stage = stage
        self.rank = stage.stage_index
        self.world_size = stage.num_stages

    def forward_backward(
        self,
        input_ids: Tensor,
        targets: Tensor,
        *,
        loss_fn: LossFunction,
        num_microbatches: int,
    ) -> Tensor | None:
        """Run one synchronous mini-batch and accumulate local stage gradients.

        ``loss_fn`` must return the mean loss for one equally-sized microbatch.
        The caller owns gradient clearing and the optimizer step.
        """

        microbatch_size = self._validate_batch(input_ids, targets, num_microbatches)
        input_microbatches = input_ids.split(microbatch_size, dim=0)
        target_microbatches = targets.split(microbatch_size, dim=0)
        microbatch_states = [
            self._forward_microbatch(input_microbatches[index], target_microbatches[index], loss_fn, num_microbatches) 
            for index in range(num_microbatches)
        ]

        for state in reversed(microbatch_states):
            self._backward_microbatch(state)

        if not self.stage.is_last_stage:
            return None
        losses = [state.loss.detach() for state in microbatch_states if state.loss is not None]
        return torch.stack(losses).sum()

    def _validate_batch(self, input_ids: Tensor, targets: Tensor, num_microbatches: int) -> int:
        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        if input_ids.shape != targets.shape:
            raise ValueError("input_ids and targets must have the same shape")
        if input_ids.ndim < 1 or input_ids.shape[0] == 0:
            raise ValueError("input_ids must have a non-empty batch dimension")
        if input_ids.shape[0] % num_microbatches:
            raise ValueError("batch size must be divisible by num_microbatches")
        if input_ids.device != self.stage.device or targets.device != self.stage.device:
            raise ValueError("input_ids, targets, and the local stage must be on the same device")
        return input_ids.shape[0] // num_microbatches

    def _forward_microbatch(
        self,
        input_ids: Tensor,
        targets: Tensor,
        loss_fn: LossFunction,
        num_microbatches: int,
    ) -> _MicrobatchState:
        input_activation = self._receive_input_activation(input_ids.shape)
        stage_input = input_ids if input_activation is None else input_activation
        output = self.stage(stage_input)

        if not self.stage.is_last_stage:
            dist.send(output.detach().contiguous(), dst=self.rank + 1)
            return _MicrobatchState(
                input_activation=input_activation,
                output=output,
                loss=None,
            )

        loss = loss_fn(output, targets)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar mean loss")
        return _MicrobatchState(
            input_activation=input_activation,
            output=output,
            loss=loss / num_microbatches,
        )

    def _receive_input_activation(self, input_shape: torch.Size) -> Tensor | None:
        if self.stage.is_first_stage:
            return None
        activation = torch.empty(
            (*input_shape, self.stage.d_model),
            device=self.stage.device,
            dtype=self.stage.activation_dtype,
        )
        dist.recv(activation, src=self.rank - 1)
        return activation.requires_grad_(True)

    def _backward_microbatch(self, state: _MicrobatchState) -> None:
        if self.stage.is_last_stage:
            if state.loss is None:
                raise RuntimeError("the last pipeline stage is missing its microbatch loss")
            state.loss.backward()
        else:
            output_gradient = torch.empty_like(state.output)
            dist.recv(output_gradient, src=self.rank + 1)
            state.output.backward(output_gradient)

        if not self.stage.is_first_stage:
            if state.input_activation is None or state.input_activation.grad is None:
                raise RuntimeError("pipeline input activation gradient is missing")
            dist.send(state.input_activation.grad.contiguous(), dst=self.rank - 1)
