#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


CASE_PATTERN = re.compile(r"cpp_n(?P<n>\d+)_d(?P<d>\d+)_(?P<mode>noncausal|causal)\.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CUDA and PyTorch attention benchmarks.")
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--pytorch-dir", type=Path, required=True)
    parser.add_argument("--before-cpp-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_cpp_log(path: Path) -> dict[str, dict[str, float]]:
    implementations: dict[str, dict[str, float]] = {}
    current: dict[str, float] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "implementation":
            current = implementations.setdefault(value, {})
        elif current is not None and key in {
            "forward_mean_ms",
            "backward_mean_ms",
            "forward_workspace_bytes",
            "backward_workspace_bytes",
        }:
            current[key] = float(value)
    return implementations


def load_results(
    cpp_dir: Path,
    pytorch_dir: Path,
) -> dict[tuple[int, int, str], dict[str, dict[str, float]]]:
    cases: dict[tuple[int, int, str], dict[str, dict[str, float]]] = {}
    for cpp_path in sorted(cpp_dir.glob("cpp_n*_d*_*.log")):
        match = CASE_PATTERN.fullmatch(cpp_path.name)
        if match is None:
            continue
        sequence_length = int(match.group("n"))
        head_dim = int(match.group("d"))
        mode = match.group("mode")
        case_key = (sequence_length, head_dim, mode)
        implementations = parse_cpp_log(cpp_path)

        pytorch_path = pytorch_dir / f"pytorch_n{sequence_length}_d{head_dim}_{mode}.json"
        payload = json.loads(pytorch_path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            implementations[result["implementation"]] = {
                "forward_mean_ms": float(result["forward_mean_ms"]),
                "backward_mean_ms": float(result["backward_mean_ms"]),
                "peak_additional_allocated_bytes": float(result["peak_additional_allocated_bytes"]),
            }
        cases[case_key] = implementations
    return cases


def format_number(value: float) -> str:
    return f"{value:.3f}"


def render_performance_table(
    cases: dict[tuple[int, int, str], dict[str, dict[str, float]]],
    mode: str,
) -> list[str]:
    lines = [
        f"## {mode}",
        "",
        "| N | naive F/B ms | tiled F/B ms | PyTorch default F/B ms | "
        "PyTorch math F/B ms | tiled speedup vs naive F/B | tiled speedup vs default F/B |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (sequence_length, _, case_mode), implementations in sorted(cases.items()):
        if case_mode != mode:
            continue
        naive = implementations["naive"]
        tiled = implementations["tiled"]
        default = implementations["pytorch_sdpa_default"]
        math_result = implementations["pytorch_sdpa_math"]
        naive_speedup = (
            naive["forward_mean_ms"] / tiled["forward_mean_ms"],
            naive["backward_mean_ms"] / tiled["backward_mean_ms"],
        )
        default_speedup = (
            default["forward_mean_ms"] / tiled["forward_mean_ms"],
            default["backward_mean_ms"] / tiled["backward_mean_ms"],
        )
        lines.append(
            f"| {sequence_length} | "
            f"{format_number(naive['forward_mean_ms'])}/{format_number(naive['backward_mean_ms'])} | "
            f"{format_number(tiled['forward_mean_ms'])}/{format_number(tiled['backward_mean_ms'])} | "
            f"{format_number(default['forward_mean_ms'])}/{format_number(default['backward_mean_ms'])} | "
            f"{format_number(math_result['forward_mean_ms'])}/{format_number(math_result['backward_mean_ms'])} | "
            f"{format_number(naive_speedup[0])}x/{format_number(naive_speedup[1])}x | "
            f"{format_number(default_speedup[0])}x/{format_number(default_speedup[1])}x |"
        )
    lines.append("")
    return lines


def render_memory_table(
    cases: dict[tuple[int, int, str], dict[str, dict[str, float]]],
) -> list[str]:
    lines = [
        "## Memory",
        "",
        "| N | naive backward workspace MiB | tiled backward workspace KiB | "
        "PyTorch default additional peak MiB | PyTorch math additional peak MiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for (sequence_length, _, mode), implementations in sorted(cases.items()):
        if mode != "noncausal":
            continue
        naive = implementations["naive"]
        tiled = implementations["tiled"]
        default = implementations["pytorch_sdpa_default"]
        math_result = implementations["pytorch_sdpa_math"]
        lines.append(
            f"| {sequence_length} | "
            f"{naive['backward_workspace_bytes'] / 2**20:.3f} | "
            f"{tiled['backward_workspace_bytes'] / 2**10:.3f} | "
            f"{default['peak_additional_allocated_bytes'] / 2**20:.3f} | "
            f"{math_result['peak_additional_allocated_bytes'] / 2**20:.3f} |"
        )
    lines.append("")
    return lines


def render_optimization_table(
    cases: dict[tuple[int, int, str], dict[str, dict[str, float]]],
    before_cpp_dir: Path,
) -> list[str]:
    lines = [
        "## Tiled optimization",
        "",
        "| N | mode | before F/B ms | final F/B ms | final speedup F/B |",
        "|---:|---|---:|---:|---:|",
    ]
    for (sequence_length, head_dim, mode), implementations in sorted(cases.items()):
        before_path = before_cpp_dir / f"cpp_n{sequence_length}_d{head_dim}_{mode}.log"
        before = parse_cpp_log(before_path)["tiled"]
        final = implementations["tiled"]
        speedup = (
            before["forward_mean_ms"] / final["forward_mean_ms"],
            before["backward_mean_ms"] / final["backward_mean_ms"],
        )
        lines.append(
            f"| {sequence_length} | {mode} | "
            f"{format_number(before['forward_mean_ms'])}/{format_number(before['backward_mean_ms'])} | "
            f"{format_number(final['forward_mean_ms'])}/{format_number(final['backward_mean_ms'])} | "
            f"{format_number(speedup[0])}x/{format_number(speedup[1])}x |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    cases = load_results(args.cpp_dir, args.pytorch_dir)
    lines = ["# FlashAttention benchmark summary", ""]
    lines.extend(render_performance_table(cases, "noncausal"))
    lines.extend(render_performance_table(cases, "causal"))
    lines.extend(render_memory_table(cases))
    if args.before_cpp_dir is not None:
        lines.extend(render_optimization_table(cases, args.before_cpp_dir))

    report = "\n".join(lines)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
