"""End-to-end benchmarks for the Assignment 1 Transformer language model."""

from __future__ import annotations

import argparse
import json
import statistics
import timeit
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


@dataclass(frozen=True)
class ModelConfig:
    """Transformer dimensions needed by ``TransformerLM``."""

    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

MODES = ("forward", "forward-backward", "full")
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
AUTOCAST_DTYPES = {
    "none": None,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class PhaseStatistics:
    """Summary statistics for one measured phase."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]


@dataclass(frozen=True)
class BenchmarkResult:
    """Configuration and phase-level timing results for one benchmark run."""

    model_size: str
    model_config: ModelConfig
    mode: str
    device: str
    device_name: str
    dtype: str
    autocast_dtype: str | None
    torch_version: str
    cuda_version: str | None
    num_cpu_threads: int
    vocab_size: int
    batch_size: int
    context_length: int
    rope_theta: float
    parameter_count: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    measurement_steps: int
    seed: int
    phases: dict[str, PhaseStatistics]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable CUDA devices early."""

    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


def validate_config(config: ModelConfig) -> None:
    """Validate constraints imposed by multi-head attention and RoPE."""

    if min(config.d_model, config.d_ff, config.num_layers, config.num_heads) <= 0:
        raise ValueError("all model dimensions must be positive")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if (config.d_model // config.num_heads) % 2 != 0:
        raise ValueError("the per-head dimension must be even for RoPE")


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work; synchronization is a no-op on CPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(operation: Callable[[], Tensor | None], device: torch.device) -> tuple[Tensor | None, float]:
    start = timeit.default_timer()
    result = operation()
    synchronize(device)
    return result, timeit.default_timer() - start


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _run_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    input_ids: Tensor,
    targets: Tensor,
    mode: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    measure: bool,
) -> dict[str, float]:
    """Run one benchmark step and optionally return synchronized phase timings."""

    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "full" and optimizer is None:
        raise ValueError("full mode requires an optimizer")

    if mode != "forward":
        model.zero_grad(set_to_none=True)

    if not measure:
        if mode == "forward":
            with torch.inference_mode(), _autocast_context(device, autocast_dtype):
                model(input_ids)
        else:
            with _autocast_context(device, autocast_dtype):
                logits = model(input_ids)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            if optimizer is not None:
                optimizer.step()
        synchronize(device)
        return {}

    synchronize(device)
    total_start = timeit.default_timer()
    timings: dict[str, float] = {}

    if mode == "forward":
        with torch.inference_mode(), _autocast_context(device, autocast_dtype):
            _, timings["forward"] = _timed_call(lambda: model(input_ids), device)
    else:
        with _autocast_context(device, autocast_dtype):
            logits, timings["forward"] = _timed_call(lambda: model(input_ids), device)
            assert logits is not None

            loss, timings["loss"] = _timed_call(
                lambda: cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)),
                device,
            )
            assert loss is not None
        _, timings["backward"] = _timed_call(loss.backward, device)

        if optimizer is not None:
            _, timings["optimizer"] = _timed_call(optimizer.step, device)

    synchronize(device)
    timings["total"] = timeit.default_timer() - total_start
    return timings


def _summarize(samples: list[float]) -> PhaseStatistics:
    samples_ms = [sample * 1_000 for sample in samples]
    return PhaseStatistics(
        mean_ms=statistics.mean(samples_ms),
        std_ms=statistics.pstdev(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        samples_ms=samples_ms,
    )


def run_benchmark(
    *,
    model_size: str,
    model_config: ModelConfig,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    autocast_dtype: torch.dtype | None,
    autocast_dtype_name: str | None,
    vocab_size: int,
    batch_size: int,
    context_length: int,
    rope_theta: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    measurement_steps: int,
    seed: int,
) -> BenchmarkResult:
    """Construct a model and benchmark synchronized end-to-end steps."""

    validate_config(model_config)
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if min(vocab_size, batch_size, context_length, measurement_steps) <= 0:
        raise ValueError("vocab_size, batch_size, context_length, and measurement_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if autocast_dtype is not None and dtype is not torch.float32:
        raise ValueError("autocast requires float32 model parameters in this benchmark")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    model.train(mode != "forward")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if mode == "full" else None
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    for _ in range(warmup_steps):
        _run_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            mode=mode,
            device=device,
            autocast_dtype=autocast_dtype,
            measure=False,
        )

    samples: dict[str, list[float]] = {}
    measurement_range = torch.cuda.nvtx.range("benchmark_measurement") if device.type == "cuda" else nullcontext()
    with measurement_range:
        for _ in range(measurement_steps):
            step_timings = _run_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                mode=mode,
                device=device,
                autocast_dtype=autocast_dtype,
                measure=True,
            )
            for phase, elapsed_seconds in step_timings.items():
                samples.setdefault(phase, []).append(elapsed_seconds)

    return BenchmarkResult(
        model_size=model_size,
        model_config=model_config,
        mode=mode,
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        dtype=dtype_name,
        autocast_dtype=autocast_dtype_name,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        num_cpu_threads=torch.get_num_threads(),
        vocab_size=vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        rope_theta=rope_theta,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        seed=seed,
        phases={phase: _summarize(values) for phase, values in samples.items()},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--d-model", type=int, help="override the selected model preset")
    parser.add_argument("--d-ff", type=int, help="override the selected model preset")
    parser.add_argument("--num-layers", type=int, help="override the selected model preset")
    parser.add_argument("--num-heads", type=int, help="override the selected model preset")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--device", default="auto", help="PyTorch device string or 'auto'")
    parser.add_argument("--dtype", choices=DTYPES, default="float32", help="model parameter and buffer storage dtype")
    parser.add_argument("--autocast-dtype", choices=AUTOCAST_DTYPES, default="none", help="optional mixed-precision compute dtype")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, help="optional path for the full result, including raw samples")
    return parser


def _resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    preset = MODEL_CONFIGS[args.model_size]
    return ModelConfig(
        d_model=args.d_model if args.d_model is not None else preset.d_model,
        d_ff=args.d_ff if args.d_ff is not None else preset.d_ff,
        num_layers=args.num_layers if args.num_layers is not None else preset.num_layers,
        num_heads=args.num_heads if args.num_heads is not None else preset.num_heads,
    )


def _print_result(result: BenchmarkResult) -> None:
    config = result.to_dict()
    phases = config.pop("phases")
    print(f"benchmark_config={json.dumps(config, sort_keys=True)}")
    for phase, stats in phases.items():
        print(f"phase={phase} mean_ms={stats['mean_ms']:.3f} std_ms={stats['std_ms']:.3f} min_ms={stats['min_ms']:.3f} max_ms={stats['max_ms']:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(
        model_size=args.model_size,
        model_config=_resolve_model_config(args),
        mode=args.mode,
        device=resolve_device(args.device),
        dtype=DTYPES[args.dtype],
        dtype_name=args.dtype,
        autocast_dtype=AUTOCAST_DTYPES[args.autocast_dtype],
        autocast_dtype_name=None if args.autocast_dtype == "none" else args.autocast_dtype,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        seed=args.seed,
    )
    _print_result(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
