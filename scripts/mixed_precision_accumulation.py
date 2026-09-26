from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch


STEPS = 1_000
INCREMENT = 0.01


@dataclass(frozen=True)
class AccumulationCase:
    name: str
    accumulator_dtype: torch.dtype
    increment_dtype: torch.dtype
    explicit_cast: bool = False


@dataclass(frozen=True)
class AccumulationResult:
    case: AccumulationCase
    stored_increment: float
    value: float
    signed_error: float
    relative_error: float
    final_ulp: float
    error_trace: tuple[float, ...]


CASES = (
    AccumulationCase(
        name="fp32_accumulator_fp32_increment",
        accumulator_dtype=torch.float32,
        increment_dtype=torch.float32,
    ),
    AccumulationCase(
        name="fp16_accumulator_fp16_increment",
        accumulator_dtype=torch.float16,
        increment_dtype=torch.float16,
    ),
    AccumulationCase(
        name="fp32_accumulator_fp16_increment_implicit_promotion",
        accumulator_dtype=torch.float32,
        increment_dtype=torch.float16,
    ),
    AccumulationCase(
        name="fp32_accumulator_fp16_increment_explicit_cast",
        accumulator_dtype=torch.float32,
        increment_dtype=torch.float16,
        explicit_cast=True,
    ),
)


def run_case(case: AccumulationCase, *, steps: int, increment: float) -> AccumulationResult:
    accumulator = torch.tensor(0, dtype=case.accumulator_dtype)
    error_trace: list[float] = []
    for step in range(1, steps + 1):
        term = torch.tensor(increment, dtype=case.increment_dtype)
        if case.explicit_cast:
            term = term.to(case.accumulator_dtype)
        accumulator += term
        error_trace.append(float(accumulator.item()) - step * increment)

    ideal = steps * increment
    value = float(accumulator.item())
    signed_error = value - ideal
    next_value = torch.nextafter(
        accumulator,
        torch.tensor(float("inf"), dtype=accumulator.dtype),
    )
    final_ulp = float((next_value - accumulator).item())
    return AccumulationResult(
        case=case,
        stored_increment=float(torch.tensor(increment, dtype=case.increment_dtype).item()),
        value=value,
        signed_error=signed_error,
        relative_error=signed_error / ideal,
        final_ulp=final_ulp,
        error_trace=tuple(error_trace),
    )


def plot_error_traces(results: list[AccumulationResult], output_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")

    import matplotlib.pyplot as plt

    colors = {
        "fp32_accumulator_fp32_increment": "#0072B2",
        "fp16_accumulator_fp16_increment": "#D55E00",
        "fp32_accumulator_fp16_increment_implicit_promotion": "#009E73",
        "fp32_accumulator_fp16_increment_explicit_cast": "#CC79A7",
    }
    labels = {
        "fp32_accumulator_fp32_increment": "FP32 acc + FP32 term",
        "fp16_accumulator_fp16_increment": "FP16 acc + FP16 term",
        "fp32_accumulator_fp16_increment_implicit_promotion": "FP32 acc + FP16 term",
        "fp32_accumulator_fp16_increment_explicit_cast": "FP32 acc + FP16 term, explicit cast",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = range(1, STEPS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for result in results:
        name = result.case.name
        line_style = "--" if result.case.explicit_cast else "-"
        axes[0].plot(steps, result.error_trace, color=colors[name], linestyle=line_style, linewidth=2, label=labels[name])
        if result.case.accumulator_dtype is torch.float32:
            axes[1].plot(steps, result.error_trace, color=colors[name], linestyle=line_style, linewidth=2, label=labels[name])

    for axis in axes:
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle=":")
        axis.set_xlabel("Accumulation step")
        axis.set_ylabel("Signed error from n * 0.01")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[0].set_title("All four cases")
    axes[1].set_title("FP32 accumulator cases (zoomed)")
    figure.suptitle("Repeated accumulation error")
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CS336 mixed-precision accumulation experiment.")
    parser.add_argument("--output-svg", type=Path, help="Optional path for the cumulative-error plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"torch_version={torch.__version__}")
    print(f"experiment_steps={STEPS}")
    print(f"mathematical_increment={INCREMENT:.17g}")
    print(f"ideal_sum={STEPS * INCREMENT:.17g}")

    results: list[AccumulationResult] = []
    for case in CASES:
        result = run_case(case, steps=STEPS, increment=INCREMENT)
        results.append(result)
        print(
            f"case={case.name} "
            f"accumulator_dtype={case.accumulator_dtype} "
            f"increment_dtype={case.increment_dtype} "
            f"explicit_cast={case.explicit_cast} "
            f"stored_increment={result.stored_increment:.17g} "
            f"result={result.value:.17g} "
            f"signed_error={result.signed_error:.17g} "
            f"relative_error={result.relative_error:.17g} "
            f"final_ulp={result.final_ulp:.17g}"
        )

    if args.output_svg is not None:
        plot_error_traces(results, args.output_svg)
        print(f"output_svg={args.output_svg}")


if __name__ == "__main__":
    main()
