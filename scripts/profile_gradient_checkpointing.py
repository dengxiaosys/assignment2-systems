from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cs336_basics.model import TransformerBlock
from cs336_systems.benchmark import MODEL_CONFIGS, ModelConfig, resolve_device
from cs336_systems.gradient_checkpointing import CheckpointStrategy, apply_checkpoint_strategy
from cs336_systems.saved_tensor_profiler import capture_saved_tensors


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss if sys.platform == "darwin" else max_rss * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_blocks(
    config: ModelConfig,
    *,
    num_layers: int,
    context_length: int,
    rope_theta: float,
    device: torch.device,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=torch.float32,
            )
            for _ in range(num_layers)
        ]
    )


def _resolve_strategy(args: argparse.Namespace) -> tuple[CheckpointStrategy, int | None]:
    if args.blocks_per_checkpoint == 0:
        return "none", None
    if args.blocks_per_checkpoint > 0:
        return "grouped", args.blocks_per_checkpoint
    if args.blocks_per_checkpoint == -1:
        return "recursive", None
    raise ValueError("blocks per checkpoint must be -1, 0, or a positive integer")


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    strategy, blocks_per_checkpoint = _resolve_strategy(args)
    device = resolve_device(args.device)
    config = MODEL_CONFIGS[args.model_size]
    num_layers = config.num_layers if args.num_layers is None else args.num_layers

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    rss_before_model = _rss_bytes()
    blocks = _build_blocks(
        config,
        num_layers=num_layers,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        device=device,
    )
    blocks.train()
    x = torch.randn(
        args.batch_size,
        args.context_length,
        config.d_model,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    gc.collect()
    _synchronize(device)
    rss_model_ready = _rss_bytes()
    peak_rss_before_forward = _peak_rss_bytes()

    forward_start = time.perf_counter()
    output, saved_tensor_profile = capture_saved_tensors(
        lambda: apply_checkpoint_strategy(
            blocks,
            x,
            strategy=strategy,
            blocks_per_checkpoint=blocks_per_checkpoint,
        ),
        tensor_roles={
            "input": (x,),
            "parameter": tuple(blocks.parameters()),
        },
    )
    loss = output.square().mean()
    _synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    rss_after_forward = _rss_bytes()
    peak_rss_after_forward = _peak_rss_bytes()

    backward_start = time.perf_counter()
    loss.backward()
    _synchronize(device)
    backward_seconds = time.perf_counter() - backward_start
    rss_after_backward = _rss_bytes()
    peak_rss_after_backward = _peak_rss_bytes()

    all_saved_metrics = saved_tensor_profile.metrics()
    non_parameter_saved_metrics = saved_tensor_profile.metrics(excluding_roles=("parameter",))
    parameter_count = sum(parameter.numel() for parameter in blocks.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in blocks.parameters())
    parameter_gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in blocks.parameters() if parameter.grad is not None)
    input_bytes = x.numel() * x.element_size()
    num_checkpoints = 0 if blocks_per_checkpoint is None else (num_layers + blocks_per_checkpoint - 1) // blocks_per_checkpoint

    result = {
        "config": {
            "model_size": args.model_size,
            "model_config": asdict(config),
            "num_layers": num_layers,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "dtype": "float32",
            "device": str(device),
            "strategy": strategy,
            "blocks_per_checkpoint": blocks_per_checkpoint,
            "num_checkpoints": num_checkpoints,
            "rope_theta": args.rope_theta,
            "seed": args.seed,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "model": {
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "parameter_gradient_bytes": parameter_gradient_bytes,
            "input_bytes": input_bytes,
        },
        "saved_tensors_after_forward": {
            "reference_count": all_saved_metrics.tensor_count,
            "logical_bytes_including_parameters": all_saved_metrics.logical_bytes,
            "logical_non_parameter_bytes": non_parameter_saved_metrics.logical_bytes,
            "unique_non_parameter_storage_bytes": non_parameter_saved_metrics.unique_storage_bytes,
        },
        "timings_seconds": {
            "forward": forward_seconds,
            "backward": backward_seconds,
            "total": forward_seconds + backward_seconds,
        },
        "memory": {
            "rss_before_model": rss_before_model,
            "rss_model_ready": rss_model_ready,
            "rss_after_forward": rss_after_forward,
            "rss_after_backward": rss_after_backward,
            "peak_rss_before_forward": peak_rss_before_forward,
            "peak_rss_after_forward": peak_rss_after_forward,
            "peak_rss_after_backward": peak_rss_after_backward,
            "peak_rss_increment_from_model_ready": max(
                0,
                peak_rss_after_backward - rss_model_ready,
            ),
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "cuda_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        },
        "numerics": {
            "loss": float(loss.detach()),
            "output_sum": float(output.detach().double().sum()),
            "input_gradient_norm": None if x.grad is None else float(x.grad.detach().double().norm()),
            "parameter_gradient_norm": float(
                torch.linalg.vector_norm(torch.stack([parameter.grad.detach().double().norm() for parameter in blocks.parameters() if parameter.grad is not None]))
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"strategy={strategy}")
    print(f"blocks_per_checkpoint={blocks_per_checkpoint}")
    print(f"parameter_count={parameter_count}")
    print(f"saved_tensor_reference_count={all_saved_metrics.tensor_count}")
    print(f"unique_non_parameter_saved_storage_bytes={non_parameter_saved_metrics.unique_storage_bytes}")
    print(f"forward_seconds={forward_seconds:.6f}")
    print(f"backward_seconds={backward_seconds:.6f}")
    print(f"peak_rss_after_backward={peak_rss_after_backward}")
    print(f"output_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile activation checkpointing over a Transformer block stack.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="large")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--blocks-per-checkpoint",
        type=int,
        required=True,
        help="-1: recursive, 0: no checkpoint, positive: non-nested group size",
    )
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_layers is not None and args.num_layers <= 0:
        raise ValueError("num layers must be positive")
    if args.batch_size <= 0 or args.context_length <= 0:
        raise ValueError("batch size and context length must be positive")
    run_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
