"""Minimal end-to-end pipeline-parallel training example."""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.pipeline_parallel import PipelineParallel, TransformerPipelineStage

WORLD_SIZE = 2
STEPS = 3
NUM_MICROBATCHES = 4
BATCH_SIZE = 8
VOCAB_SIZE = 256
CONTEXT_LENGTH = 32
D_MODEL = 64


def language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def train_worker(rank: int, port: int, backend: str) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60))
    dist.init_process_group(backend, store=store, rank=rank, world_size=WORLD_SIZE)

    try:
        torch.manual_seed(20261003 + rank)

        # Each rank owns only one contiguous part of the Transformer.
        stage = TransformerPipelineStage.from_dimensions(
            stage_index=rank,
            num_stages=WORLD_SIZE,
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            d_model=D_MODEL,
            num_layers=4,
            num_heads=4,
            d_ff=128,
            rope_theta=10_000.0,
            device=device,
            dtype=torch.float32,
        )
        pipeline = PipelineParallel(stage)
        optimizer = AdamW(stage.parameters(), lr=1e-3)
        print(
            f"rank={rank} layers=[{stage.partition.start},{stage.partition.end}) parameter_count={sum(parameter.numel() for parameter in stage.parameters())}",
            flush=True,
        )

        # The first stage consumes token IDs; the last stage consumes targets.
        generator = torch.Generator(device=device).manual_seed(42)
        for step in range(STEPS):
            shape = (BATCH_SIZE, CONTEXT_LENGTH)
            input_ids = torch.randint(VOCAB_SIZE, shape, generator=generator, device=device)
            targets = torch.randint(VOCAB_SIZE, shape, generator=generator, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss = pipeline.forward_backward(
                input_ids,
                targets,
                loss_fn=language_model_loss,
                num_microbatches=NUM_MICROBATCHES,
            )
            optimizer.step()

            # Only the last stage computes the language-model loss.
            if loss is not None:
                print(f"step={step} loss={loss.item():.6f}", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    use_nccl = dist.is_nccl_available() and torch.cuda.device_count() >= WORLD_SIZE
    backend = "nccl" if use_nccl else "gloo"
    server = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    mp.spawn(
        train_worker,
        args=(server.port, backend),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
