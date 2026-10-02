"""Run a reproducible CPU/Gloo optimizer-state sharding experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

from cs336_systems.sharded_optimizer import ShardedOptimizer


@dataclass(frozen=True)
class ExperimentConfig:
    world_size: int
    steps: int
    width: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    num_threads: int
    master_port: int
    output: Path


class FourLayerMLP(nn.Module):
    """Equal-sized matrices make the expected two-rank state split explicit."""

    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(4)])

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return hidden


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _optimizer_state_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(_tensor_bytes(value) for state in optimizer.state.values() for value in state.values() if isinstance(value, Tensor))


def _worker(rank: int, config: ExperimentConfig) -> None:
    torch.set_num_threads(config.num_threads)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group("gloo", rank=rank, world_size=config.world_size)
    try:
        torch.manual_seed(config.seed)
        initial_model = FourLayerMLP(config.width)
        baseline_model = deepcopy(initial_model)
        sharded_model = deepcopy(initial_model)
        baseline_optimizer = torch.optim.AdamW(
            baseline_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        sharded_optimizer = ShardedOptimizer(
            sharded_model.parameters(),
            torch.optim.AdamW,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        maximum_baseline_error = 0.0
        for step in range(config.steps):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed + step + 1)
            inputs = torch.randn(config.batch_size, config.width, generator=generator)
            targets = torch.randn(config.batch_size, config.width, generator=generator)

            baseline_optimizer.zero_grad(set_to_none=True)
            sharded_optimizer.zero_grad(set_to_none=True)
            baseline_loss = (baseline_model(inputs) - targets).square().mean()
            sharded_loss = (sharded_model(inputs) - targets).square().mean()
            baseline_loss.backward()
            sharded_loss.backward()
            baseline_optimizer.step()
            sharded_optimizer.step()

            for baseline_parameter, sharded_parameter in zip(
                baseline_model.parameters(),
                sharded_model.parameters(),
                strict=True,
            ):
                error = (baseline_parameter - sharded_parameter).abs().max().item()
                maximum_baseline_error = max(maximum_baseline_error, error)

        sharded_parameters = tuple(sharded_model.parameters())
        maximum_cross_rank_error = 0.0
        for parameter in sharded_parameters:
            rank_zero_parameter = parameter.detach().clone()
            dist.broadcast(rank_zero_parameter, src=0)
            maximum_cross_rank_error = max(
                maximum_cross_rank_error,
                (parameter - rank_zero_parameter).abs().max().item(),
            )

        local_optimizer = sharded_optimizer.local_optimizer
        if local_optimizer is None:
            local_state_entries = 0
            local_state_tensor_bytes = 0
        else:
            local_state_entries = len(local_optimizer.state)
            local_state_tensor_bytes = _optimizer_state_tensor_bytes(local_optimizer)
        local_owned_parameters = [
            parameter
            for parameter, owner in zip(
                sharded_parameters,
                sharded_optimizer.parameter_owners,
                strict=True,
            )
            if owner == rank
        ]
        local_result = {
            "rank": rank,
            "owner_sequence": list(sharded_optimizer.parameter_owners),
            "owned_parameter_count": len(local_owned_parameters),
            "owned_parameter_elements": sum(parameter.numel() for parameter in local_owned_parameters),
            "owned_parameter_bytes": sum(_tensor_bytes(parameter) for parameter in local_owned_parameters),
            "optimizer_state_entries": local_state_entries,
            "optimizer_state_tensor_bytes": local_state_tensor_bytes,
            "maximum_baseline_parameter_error": maximum_baseline_error,
            "maximum_cross_rank_parameter_error": maximum_cross_rank_error,
        }
        rank_results: list[dict[str, Any] | None] = [None] * config.world_size
        dist.all_gather_object(rank_results, local_result)

        if rank == 0:
            complete_rank_results = [rank_result for rank_result in rank_results if rank_result is not None]
            if len(complete_rank_results) != config.world_size:
                raise RuntimeError("did not receive an experiment result from every rank")
            global_maximum_baseline_error = max(float(rank_result["maximum_baseline_parameter_error"]) for rank_result in complete_rank_results)
            global_maximum_cross_rank_error = max(float(rank_result["maximum_cross_rank_parameter_error"]) for rank_result in complete_rank_results)
            model_parameter_bytes = sum(_tensor_bytes(parameter) for parameter in sharded_parameters)
            baseline_state_bytes = _optimizer_state_tensor_bytes(baseline_optimizer)
            result = {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "environment": {
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "platform": platform.platform(),
                    "backend": dist.get_backend(),
                },
                "config": {
                    **asdict(config),
                    "output": str(config.output),
                },
                "model": {
                    "parameter_count": len(sharded_parameters),
                    "parameter_elements": sum(parameter.numel() for parameter in sharded_parameters),
                    "parameter_bytes_per_rank": model_parameter_bytes,
                },
                "baseline": {
                    "optimizer_state_entries_per_rank": len(baseline_optimizer.state),
                    "optimizer_state_tensor_bytes_per_rank": baseline_state_bytes,
                },
                "sharded": {
                    "broadcasts_per_step": len(sharded_parameters),
                    "broadcast_payload_bytes_per_step": model_parameter_bytes,
                    "ranks": complete_rank_results,
                    "total_optimizer_state_tensor_bytes_across_ranks": sum(rank_result["optimizer_state_tensor_bytes"] for rank_result in complete_rank_results),
                },
                "correctness": {
                    "global_maximum_baseline_parameter_error": global_maximum_baseline_error,
                    "global_maximum_cross_rank_parameter_error": global_maximum_cross_rank_error,
                },
                "status": "passed" if global_maximum_baseline_error == 0.0 and global_maximum_cross_rank_error == 0.0 else "failed",
            }
            config.output.parent.mkdir(parents=True, exist_ok=True)
            config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"experiment_status={result['status']}")
            print(f"result_path={config.output}")
            print(f"model_parameter_bytes_per_rank={model_parameter_bytes}")
            print(f"baseline_optimizer_state_tensor_bytes_per_rank={baseline_state_bytes}")
            print(f"global_maximum_baseline_parameter_error={global_maximum_baseline_error}")
            print(f"global_maximum_cross_rank_parameter_error={global_maximum_cross_rank_error}")
            for rank_result in complete_rank_results:
                print(
                    f"rank={rank_result['rank']} "
                    f"owned_parameter_bytes={rank_result['owned_parameter_bytes']} "
                    f"optimizer_state_tensor_bytes={rank_result['optimizer_state_tensor_bytes']} "
                    f"maximum_baseline_parameter_error={rank_result['maximum_baseline_parameter_error']} "
                    f"maximum_cross_rank_parameter_error={rank_result['maximum_cross_rank_parameter_error']}"
                )
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/sharded_optimizer/cpu_gloo_correctness.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("world_size", "steps", "width", "batch_size", "num_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    config = ExperimentConfig(
        world_size=args.world_size,
        steps=args.steps,
        width=args.width,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_threads=args.num_threads,
        master_port=_available_port(),
        output=args.output,
    )
    mp.spawn(_worker, args=(config,), nprocs=config.world_size, join=True)
    result = json.loads(config.output.read_text(encoding="utf-8"))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
