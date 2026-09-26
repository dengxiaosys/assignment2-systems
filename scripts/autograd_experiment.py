from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cs336_systems.rmsnorm_fusion import (
    RMSNormResult,
    RMSNormVariant,
    RMSNormFusionWorkload,
    compare_rmsnorm_results,
)
from cs336_systems.saved_tensor_profiler import SavedTensorProfile, capture_saved_tensors


def _profile_variant(
    input_data: Tensor,
    *,
    eps: float,
    variant: RMSNormVariant,
    compile_backend: str | None,
) -> tuple[dict[str, Any], RMSNormResult]:
    workload = RMSNormFusionWorkload(
        hidden_size=input_data.shape[-1],
        eps=eps,
        variant=variant,
        compile_backend=compile_backend,
        device=input_data.device,
    )
    workload.warm_up(input_data)
    x = input_data.detach().clone().requires_grad_(True)
    result, profile = capture_saved_tensors(
        lambda: workload.run(x),
        tensor_roles={
            "input": (x,),
            "parameter": workload.parameter_tensors,
        },
    )
    return _summarize_profile(profile, variant=variant, input_shape=tuple(x.shape)), result


def _summarize_profile(
    profile: SavedTensorProfile,
    *,
    variant: RMSNormVariant,
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    non_parameter_metrics = profile.metrics(excluding_roles=("parameter",))
    full_size_non_parameter_save_count = sum(event.role != "parameter" and event.shape == input_shape for event in profile.saved_events)
    return {
        "variant": variant,
        **profile.to_dict(),
        "logical_non_parameter_saved_bytes": non_parameter_metrics.logical_bytes,
        "unique_non_parameter_saved_storage_bytes": non_parameter_metrics.unique_storage_bytes,
        "full_size_non_parameter_save_count": full_size_non_parameter_save_count,
    }


def _resolve_variants(mode: str) -> tuple[RMSNormVariant, ...]:
    if mode == "both":
        return "eager", "compiled"
    if mode == "eager":
        return ("eager",)
    if mode == "compiled":
        return ("compiled",)
    raise ValueError(f"unsupported mode: {mode}")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    input_data = torch.randn(
        args.batch_size,
        args.context_length,
        args.hidden_size,
        dtype=torch.float32,
    )
    measured_runs = {
        variant: _profile_variant(
            input_data,
            eps=args.eps,
            variant=variant,
            compile_backend=args.compile_backend,
        )
        for variant in _resolve_variants(args.mode)
    }
    summaries = {variant: measured[0] for variant, measured in measured_runs.items()}
    results = {variant: measured[1] for variant, measured in measured_runs.items()}

    result: dict[str, Any] = {
        "config": {
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "hidden_size": args.hidden_size,
            "eps": args.eps,
            "mode": args.mode,
            "compile_backend": args.compile_backend,
            "seed": args.seed,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "variants": summaries,
    }
    if set(results) == {"eager", "compiled"}:
        result["correctness"] = compare_rmsnorm_results(
            results["eager"],
            results["compiled"],
        ).to_dict()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_result(result)
    return result


def _print_result(result: dict[str, Any]) -> None:
    for variant, summary in result["variants"].items():
        for event in summary["events"]:
            print(
                f"variant={variant} phase={event['phase']} save_index={event['save_index']} "
                f"storage_index={event['storage_index']} role={event['role']} shape={event['shape']} "
                f"dtype={event['dtype']} tensor_nbytes={event['tensor_nbytes']} grad_fn={event['grad_fn']}"
            )
        print(
            f"variant={variant} saved_tensor_count={summary['saved_tensor_count']} "
            f"logical_saved_bytes={summary['logical_saved_bytes']} "
            f"unique_saved_storage_bytes={summary['unique_saved_storage_bytes']} "
            f"full_size_non_parameter_save_count={summary['full_size_non_parameter_save_count']}"
        )
    for name, comparison in result.get("correctness", {}).items():
        print(f"comparison={name} allclose={comparison['allclose']} max_abs_diff={comparison['max_abs_diff']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare saved tensors from eager and compiled RMSNorm.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--mode", choices=("eager", "compiled", "both"), default="both")
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend; pass 'default' to omit the backend argument",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/autograd_residuals/rmsnorm_saved_tensors.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.context_length <= 0 or args.hidden_size <= 0:
        raise ValueError("batch size, context length, and hidden size must be positive")
    if args.compile_backend == "default":
        args.compile_backend = None
    result = run_experiment(args)
    print(f"output_json={args.output_json}")
    if "correctness" in result and not all(comparison["allclose"] for comparison in result["correctness"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
