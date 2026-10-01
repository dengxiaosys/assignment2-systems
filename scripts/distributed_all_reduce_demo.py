from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class ExperimentConfig:
    world_size: int
    vector_size: int
    seed: int
    master_addr: str
    master_port: int
    timeout_seconds: int
    torch_threads: int


def make_rank_data(rank: int, *, vector_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank)
    return torch.randint(0, 10, (vector_size,), dtype=torch.int64, generator=generator)


def expected_sum(config: ExperimentConfig) -> torch.Tensor:
    rank_tensors = [
        make_rank_data(rank, vector_size=config.vector_size, seed=config.seed)
        for rank in range(config.world_size)
    ]
    return torch.stack(rank_tensors).sum(dim=0)


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def setup_process_group(rank: int, config: ExperimentConfig) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=config.world_size,
        timeout=timedelta(seconds=config.timeout_seconds),
    )


def run_worker(rank: int, config: ExperimentConfig) -> None:
    torch.set_num_threads(config.torch_threads)
    setup_process_group(rank, config)
    try:
        data = make_rank_data(rank, vector_size=config.vector_size, seed=config.seed)
        expected = expected_sum(config)

        print(f"rank={rank} phase=before_all_reduce data={data.tolist()}", flush=True)
        storage_pointer = data.data_ptr()
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

        correct = torch.equal(data, expected)
        in_place = data.data_ptr() == storage_pointer
        print(
            f"rank={rank} phase=after_all_reduce data={data.tolist()} "
            f"expected={expected.tolist()} correct={correct} in_place={in_place}",
            flush=True,
        )
        if not correct or not in_place:
            raise RuntimeError(f"rank={rank} failed all-reduce validation")
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic CPU Gloo all-reduce demonstration.")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--vector-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="Rendezvous port; 0 selects an available local port.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    if args.vector_size <= 0:
        raise ValueError("vector size must be positive")
    if not 0 <= args.master_port <= 65535:
        raise ValueError("master port must be between 0 and 65535")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout seconds must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")

    master_port = args.master_port or find_available_port()
    config = ExperimentConfig(
        world_size=args.world_size,
        vector_size=args.vector_size,
        seed=args.seed,
        master_addr=args.master_addr,
        master_port=master_port,
        timeout_seconds=args.timeout_seconds,
        torch_threads=args.torch_threads,
    )

    print("experiment_backend=gloo")
    print(f"experiment_world_size={config.world_size}")
    print(f"experiment_vector_size={config.vector_size}")
    print(f"experiment_seed={config.seed}")
    print(f"experiment_master_addr={config.master_addr}")
    print(f"experiment_master_port={config.master_port}")
    print(f"experiment_torch_threads_per_worker={config.torch_threads}")
    mp.spawn(run_worker, args=(config,), nprocs=config.world_size, join=True)
    print("experiment_status=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
