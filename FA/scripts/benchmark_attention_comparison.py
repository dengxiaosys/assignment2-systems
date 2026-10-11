"""Alternate custom/efficient timing order on identical inputs, with FP64 row checks."""

import argparse
import json
from pathlib import Path

import torch

from src.benchmark_measurement import measure_latency
from src.cuda_fa2_forward import implementation_metadata
from src.forward_registry import get_attention_forward, validate_implementation
from src.numerical_verification import REFERENCE_TOLERANCE, reference_row


def sampled_error(forward, inputs, causal):
    cpu_inputs = [tensor.cpu().double() for tensor in inputs]
    output = forward(*inputs, is_causal=causal).cpu().double()
    seq_len = inputs[0].shape[0]
    indices = sorted(qi for qi in {0, 1, 31, 32, 63, 64, seq_len // 2, seq_len - 1} if qi < seq_len)
    records = []
    for qi in indices:
        expected = reference_row(*cpu_inputs, qi, causal)
        torch.testing.assert_close(
            output[qi], expected, rtol=REFERENCE_TOLERANCE, atol=REFERENCE_TOLERANCE,
        )
        records.append({"query_index": qi, "max_abs_error": (output[qi] - expected).abs().max().item()})
    return records


def compare_case(shape, causal, seed, rounds, iterations, warmup):
    torch.manual_seed(seed)
    inputs = [torch.randn(*shape, device="cuda") for _ in range(3)]
    implementations = ("cuda_fa2_v4", "efficient")
    forwards = {name: get_attention_forward(name) for name in implementations}
    checks = {}
    for name, forward in forwards.items():
        validate_implementation(name, *inputs, is_causal=causal)
        checks[name] = sampled_error(forward, inputs, causal)
    samples = []
    for roundi in range(rounds):
        order = implementations if roundi % 2 == 0 else implementations[::-1]
        pair = {"round": roundi, "order": list(order)}
        for name in order:
            operation = lambda: forwards[name](*inputs, is_causal=causal)
            pair[name] = measure_latency(operation, torch.device("cuda"), warmup, iterations, 1)
        pair["efficient_over_custom"] = (
            pair["efficient"]["cuda_event_ms_mean"] / pair["cuda_fa2_v4"]["cuda_event_ms_mean"]
        )
        samples.append(pair)
    return {
        "shape": list(shape), "causal": causal, "seed": seed, "rounds": samples,
        "verified_all_rows": False, "sampled_cpu_fp64": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("paired", "matrix"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.suite == "paired":
        cases = [((16384, 64), causal, seed) for seed in (0, 1, 7) for causal in (False, True)]
        rounds, iterations, warmup = 5, 20, 5
    else:
        shapes = [(s, d) for s in (512, 1024, 4096, 8192, 16384) for d in (32, 64, 96, 128)]
        shapes += [(513, d) for d in (31, 33, 65, 97)]
        cases = [(shape, causal, 0) for shape in shapes for causal in (False, True)]
        rounds, iterations, warmup = 3, 10, 5
    payload = {
        "suite": args.suite, **implementation_metadata("cuda_fa2_v4"),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "dtype": "float32", "tf32": False, "cpu_threads": 1,
        "warmup_per_batch": warmup, "iterations_per_batch": iterations,
        "rtol": REFERENCE_TOLERANCE, "atol": REFERENCE_TOLERANCE,
        "reference": "CPU FP64 sampled rows; no full S x S attention matrix",
        "cases": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for shape, causal, seed in cases:
        case = compare_case(shape, causal, seed, rounds, iterations, warmup)
        payload["cases"].append(case)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            f"{shape} causal={causal} seed={seed} "
            f"ratios={[round(pair['efficient_over_custom'], 4) for pair in case['rounds']]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
