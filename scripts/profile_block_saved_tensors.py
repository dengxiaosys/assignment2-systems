from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cs336_basics.model import TransformerBlock
from cs336_systems.benchmark import DTYPES, MODEL_CONFIGS
from cs336_systems.saved_tensor_profiler import capture_saved_tensors


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _autocast_context(dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cpu", dtype=dtype)


def _source_location(_tensor: Tensor) -> str:
    for frame in reversed(traceback.extract_stack(limit=40)):
        if frame.filename.endswith(("cs336_basics/model.py", "cs336_basics/nn_utils.py")):
            return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
    return "pytorch_internal"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    config = MODEL_CONFIGS[args.model_size]
    model_dtype = DTYPES[args.dtype]
    autocast_dtype = None if args.autocast_dtype == "none" else DTYPES[args.autocast_dtype]
    if autocast_dtype is not None and model_dtype is not torch.float32:
        raise ValueError("autocast requires float32 model parameters in this profiler")

    block = TransformerBlock(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
        device="cpu",
        dtype=model_dtype,
    )
    block.train()
    x = torch.randn(
        args.batch_size,
        args.context_length,
        config.d_model,
        dtype=model_dtype,
        requires_grad=True,
    )

    rss_before_forward = _rss_bytes()
    rss_after_forward: int | None = None

    def forward_backward() -> None:
        nonlocal rss_after_forward
        with _autocast_context(autocast_dtype):
            output = block(x)
            loss = output.float().square().mean()
        rss_after_forward = _rss_bytes()
        loss.backward()

    _, profile = capture_saved_tensors(
        forward_backward,
        tensor_roles={
            "input": (x,),
            "parameter": tuple(block.parameters()),
        },
        source_resolver=_source_location,
    )
    rss_after_backward = _rss_bytes()
    if rss_after_forward is None:
        raise RuntimeError("forward did not produce an RSS sample")

    saved_references = [
        {
            "shape": list(event.shape),
            "dtype": event.dtype,
            "tensor_nbytes": event.tensor_nbytes,
            "storage_nbytes": event.storage_nbytes,
            "source": "pytorch_internal" if event.source is None else event.source,
            "is_parameter_storage": event.role == "parameter",
        }
        for event in profile.saved_events
    ]

    source_bytes: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for record in saved_references:
        if bool(record["is_parameter_storage"]):
            continue
        source = str(record["source"])
        source_bytes[source] += int(record["tensor_nbytes"])
        source_counts[source] += 1
    total_logical_non_parameter_bytes = sum(source_bytes.values())
    source_summary = [
        {
            "source": source,
            "count": source_counts[source],
            "logical_saved_bytes": size_bytes,
            "logical_saved_percent": 100 * size_bytes / total_logical_non_parameter_bytes,
        }
        for source, size_bytes in sorted(source_bytes.items(), key=lambda item: item[1], reverse=True)
    ]
    non_parameter_references = [record for record in saved_references if not bool(record["is_parameter_storage"])]
    largest_references = sorted(
        non_parameter_references,
        key=lambda record: int(record["tensor_nbytes"]),
        reverse=True,
    )[:20]
    parameter_gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in block.parameters() if parameter.grad is not None)

    result = {
        "config": {
            "model_size": args.model_size,
            "model_config": asdict(config),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "dtype": args.dtype,
            "autocast_dtype": None if autocast_dtype is None else args.autocast_dtype,
            "seed": args.seed,
            "torch_version": str(torch.__version__),
        },
        "block_parameter_count": sum(parameter.numel() for parameter in block.parameters()),
        "block_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in block.parameters()),
        "parameter_gradient_bytes": parameter_gradient_bytes,
        "input_gradient_bytes": 0 if x.grad is None else x.grad.numel() * x.grad.element_size(),
        "saved_tensor_reference_count": len(saved_references),
        "non_parameter_saved_tensor_reference_count": len(non_parameter_references),
        "logical_saved_tensor_bytes_including_parameters": profile.metrics().logical_bytes,
        "logical_non_parameter_saved_tensor_bytes": total_logical_non_parameter_bytes,
        "unique_non_parameter_storage_bytes": profile.metrics(excluding_roles=("parameter",)).unique_storage_bytes,
        "rss_bytes": {
            "before_forward": rss_before_forward,
            "after_forward": rss_after_forward,
            "after_backward": rss_after_backward,
        },
        "source_summary": source_summary,
        "largest_non_parameter_saved_tensor_references": largest_references,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"block_parameter_count={result['block_parameter_count']}")
    print(f"logical_non_parameter_saved_tensor_bytes={total_logical_non_parameter_bytes}")
    print(f"unique_non_parameter_storage_bytes={result['unique_non_parameter_storage_bytes']}")
    print(f"parameter_gradient_bytes={parameter_gradient_bytes}")
    for index, item in enumerate(source_summary[:5], start=1):
        print(f"top_source_rank={index} source={item['source']} logical_saved_bytes={item['logical_saved_bytes']} logical_saved_percent={item['logical_saved_percent']:.3f}")
    print(f"output_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="large")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--autocast-dtype", choices=("none", "bfloat16"), default="none")
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.context_length <= 0:
        raise ValueError("batch size and context length must be positive")
    run_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
