from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import DTYPES, MODEL_CONFIGS, resolve_device


@dataclass(frozen=True)
class MemorySample:
    elapsed_ms: float
    stage: str
    rss_bytes: int
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    step: int = 0


class MemoryRecorder:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[MemorySample] = []
        self._stage = "baseline"
        self._step = 0
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_rss_bytes() -> int:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    def _capture_unlocked(self) -> None:
        cuda_allocated = None
        cuda_reserved = None
        if self.device.type == "cuda":
            cuda_allocated = torch.cuda.memory_allocated(self.device)
            cuda_reserved = torch.cuda.memory_reserved(self.device)
        self.samples.append(
            MemorySample(
                elapsed_ms=(time.perf_counter() - self._start_time) * 1_000,
                stage=self._stage,
                rss_bytes=self._read_rss_bytes(),
                cuda_allocated_bytes=cuda_allocated,
                cuda_reserved_bytes=cuda_reserved,
                step=self._step,
            )
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                self._capture_unlocked()

    def start(self) -> None:
        self._start_time = time.perf_counter()
        with self._lock:
            self._capture_unlocked()
        self._thread = threading.Thread(target=self._sample_loop, name="memory-recorder", daemon=True)
        self._thread.start()

    def transition(self, stage: str, step: int | None = None) -> None:
        with self._lock:
            self._capture_unlocked()
            if step is not None:
                self._step = step
            self._stage = stage
            self._capture_unlocked()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._capture_unlocked()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _summarize_samples(samples: list[MemorySample]) -> dict[str, Any]:
    stage_peaks: dict[str, dict[str, int | None]] = {}
    for stage in dict.fromkeys(sample.stage for sample in samples):
        stage_samples = [sample for sample in samples if sample.stage == stage]
        cuda_allocated_values = [sample.cuda_allocated_bytes for sample in stage_samples if sample.cuda_allocated_bytes is not None]
        cuda_reserved_values = [sample.cuda_reserved_bytes for sample in stage_samples if sample.cuda_reserved_bytes is not None]
        stage_peaks[stage] = {
            "rss_bytes": max(sample.rss_bytes for sample in stage_samples),
            "cuda_allocated_bytes": max(cuda_allocated_values) if cuda_allocated_values else None,
            "cuda_reserved_bytes": max(cuda_reserved_values) if cuda_reserved_values else None,
        }
    cuda_allocated_values = [sample.cuda_allocated_bytes for sample in samples if sample.cuda_allocated_bytes is not None]
    cuda_reserved_values = [sample.cuda_reserved_bytes for sample in samples if sample.cuda_reserved_bytes is not None]
    step_stage_peaks: dict[str, dict[str, dict[str, int | None]]] = {}
    for step in sorted({sample.step for sample in samples if sample.step > 0}):
        step_samples = [sample for sample in samples if sample.step == step]
        step_stage_peaks[str(step)] = {}
        for stage in dict.fromkeys(sample.stage for sample in step_samples):
            matching_samples = [sample for sample in step_samples if sample.stage == stage]
            matching_allocated = [sample.cuda_allocated_bytes for sample in matching_samples if sample.cuda_allocated_bytes is not None]
            matching_reserved = [sample.cuda_reserved_bytes for sample in matching_samples if sample.cuda_reserved_bytes is not None]
            step_stage_peaks[str(step)][stage] = {
                "rss_bytes": max(sample.rss_bytes for sample in matching_samples),
                "cuda_allocated_bytes": max(matching_allocated) if matching_allocated else None,
                "cuda_reserved_bytes": max(matching_reserved) if matching_reserved else None,
            }
    return {
        "peak_rss_bytes": max(sample.rss_bytes for sample in samples),
        "peak_cuda_allocated_bytes": max(cuda_allocated_values) if cuda_allocated_values else None,
        "peak_cuda_reserved_bytes": max(cuda_reserved_values) if cuda_reserved_values else None,
        "stage_peaks": stage_peaks,
        "step_stage_peaks": step_stage_peaks,
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model_dtype = DTYPES[args.dtype]
    autocast_dtype = None if args.autocast_dtype == "none" else DTYPES[args.autocast_dtype]
    config = MODEL_CONFIGS[args.model_size]
    if autocast_dtype is not None and model_dtype is not torch.float32:
        raise ValueError("autocast requires float32 model parameters in this profiler")
    if args.output_snapshot is not None and device.type != "cuda":
        raise ValueError("--output-snapshot requires a CUDA device")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=model_dtype,
    )
    model.train(args.mode == "full")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) if args.mode == "full" else None
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if args.output_snapshot is not None:
            torch.cuda.memory._record_memory_history(max_entries=args.snapshot_max_entries)

    recorder = MemoryRecorder(device, args.sample_interval_ms / 1_000)
    recorder.start()
    step_timings_ms: list[dict[str, float]] = []

    for step in range(1, args.steps + 1):
        timings_ms: dict[str, float] = {}
        if optimizer is not None:
            recorder.transition("zero_grad", step)
            optimizer.zero_grad(set_to_none=True)
        recorder.transition("forward", step)
        start = time.perf_counter()
        if args.mode == "forward":
            with torch.inference_mode(), _autocast_context(device, autocast_dtype):
                logits = model(input_ids)
        else:
            with _autocast_context(device, autocast_dtype):
                logits = model(input_ids)
        _synchronize(device)
        timings_ms["forward"] = (time.perf_counter() - start) * 1_000

        if args.mode == "full":
            recorder.transition("loss", step)
            start = time.perf_counter()
            with _autocast_context(device, autocast_dtype):
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            _synchronize(device)
            timings_ms["loss"] = (time.perf_counter() - start) * 1_000

            recorder.transition("backward", step)
            start = time.perf_counter()
            loss.backward()
            _synchronize(device)
            timings_ms["backward"] = (time.perf_counter() - start) * 1_000

            recorder.transition("optimizer", step)
            start = time.perf_counter()
            assert optimizer is not None
            optimizer.step()
            _synchronize(device)
            timings_ms["optimizer"] = (time.perf_counter() - start) * 1_000

        recorder.transition("complete", step)
        step_timings_ms.append(timings_ms)
        del logits
        if args.mode == "full":
            del loss

    recorder.stop()

    if args.output_snapshot is not None:
        args.output_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.output_snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)

    summary = _summarize_samples(recorder.samples)
    result = {
        "config": {
            "model_size": args.model_size,
            "model_config": asdict(config),
            "mode": args.mode,
            "device": str(device),
            "dtype": args.dtype,
            "autocast_dtype": None if autocast_dtype is None else args.autocast_dtype,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "steps": args.steps,
            "sample_interval_ms": args.sample_interval_ms,
            "torch_version": str(torch.__version__),
            "num_cpu_threads": torch.get_num_threads(),
            "seed": args.seed,
        },
        "step_timings_ms": step_timings_ms,
        "summary": summary,
        "samples": [asdict(sample) for sample in recorder.samples],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"parameter_count={parameter_count}")
    print(f"parameter_bytes={parameter_bytes}")
    print(f"peak_rss_bytes={summary['peak_rss_bytes']}")
    print(f"peak_cuda_allocated_bytes={summary['peak_cuda_allocated_bytes']}")
    print(f"step_timings_ms={json.dumps(step_timings_ms, sort_keys=True)}")
    print(f"output_json={args.output_json}")
    if args.output_snapshot is not None:
        print(f"output_snapshot={args.output_snapshot}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="large")
    parser.add_argument("--mode", choices=("forward", "full"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--autocast-dtype", choices=("none", "bfloat16", "float16"), default="none")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-snapshot", type=Path)
    parser.add_argument("--snapshot-max-entries", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.context_length <= 0 or args.steps <= 0:
        raise ValueError("batch size, context length, and steps must be positive")
    if args.sample_interval_ms <= 0:
        raise ValueError("sample interval must be positive")
    run_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
