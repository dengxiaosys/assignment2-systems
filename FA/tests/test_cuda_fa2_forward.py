"""Custom C++/CUDA FA2-style forward tests."""

import argparse
import json
import unittest
from pathlib import Path

import torch

from src.cuda_fa2_forward import (
    cuda_fa2_attention_forward,
    implementation_metadata,
)
from src.forward_registry import get_attention_forward, validate_implementation
from src.numerical_verification import REFERENCE_TOLERANCE, reference_row, verify_against_cpu_fp64


DEVICE = torch.device("cpu")
IMPLEMENTATION = "cuda_fa2"
FORWARD = cuda_fa2_attention_forward
RUN_LONG_SEQUENCE = False


def validate_cuda_fa2_support(q, k, v, *, is_causal=False):
    validate_implementation(IMPLEMENTATION, q, k, v, is_causal=is_causal)


class CudaFa2ContractTests(unittest.TestCase):
    def test_rejects_cpu_before_loading_extension(self):
        q = torch.randn(7, 16)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            validate_cuda_fa2_support(q, q, q)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            FORWARD(q, q, q)


class CudaFa2NumericalTests(unittest.TestCase):
    def setUp(self):
        if DEVICE.type != "cuda":
            self.skipTest("real custom CUDA validation requires --device cuda")
        torch.manual_seed(0)

    def test_matches_cpu_fp64_reference(self):
        cases = [(1, 1), (7, 3), (65, 64), (129, 96), (257, 128)]
        for seq_len, head_dim in cases:
            for causal in (False, True):
                with self.subTest(
                    seq_len=seq_len,
                    head_dim=head_dim,
                    causal=causal,
                ):
                    inputs = [
                        torch.randn(
                            seq_len,
                            head_dim,
                            device=DEVICE,
                            dtype=torch.float32,
                            requires_grad=True,
                        )
                        for _ in range(3)
                    ]
                    copies = [tensor.detach().clone() for tensor in inputs]
                    validate_cuda_fa2_support(
                        *inputs,
                        is_causal=causal,
                    )
                    verify_against_cpu_fp64(
                        FORWARD,
                        *inputs,
                        is_causal=causal,
                    )
                    output = FORWARD(
                        *inputs,
                        is_causal=causal,
                    )
                    self.assertEqual(output.shape, inputs[0].shape)
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertEqual(output.device, inputs[0].device)
                    self.assertFalse(output.requires_grad)
                    for tensor, copy in zip(inputs, copies):
                        torch.testing.assert_close(tensor, copy)

    def test_all_supported_dimensions(self):
        for head_dim in range(1, 129):
            inputs = [torch.randn(5, head_dim, device=DEVICE) for _ in range(3)]
            for causal in (False, True):
                with self.subTest(head_dim=head_dim, causal=causal):
                    verify_against_cpu_fp64(FORWARD, *inputs, is_causal=causal)

    def test_query_tails_and_strip_boundaries(self):
        cases = [(s, 64) for s in (1, 2, 3, 4, 5, 7, 65)]
        cases += [(7, d) for d in (31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128)]
        for seq_len, head_dim in cases:
            for seed in (1, 7):
                torch.manual_seed(seed)
                inputs = [torch.randn(seq_len, head_dim, device=DEVICE) for _ in range(3)]
                for causal in (False, True):
                    with self.subTest(shape=(seq_len, head_dim), seed=seed, causal=causal):
                        verify_against_cpu_fp64(FORWARD, *inputs, is_causal=causal)

    def test_uniform_attention_and_single_key(self):
        q, k, v = [torch.randn(7, 33, device=DEVICE) for _ in range(3)]
        q.zero_()
        for causal in (False, True):
            expected = v.cumsum(0) / torch.arange(1, 8, device=DEVICE)[:, None] if causal else v.mean(0).expand_as(v)
            torch.testing.assert_close(FORWARD(q, k, v, is_causal=causal), expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(FORWARD(q[:1], k[:1], v[:1], is_causal=True), v[:1])

    def test_future_keys_do_not_change_causal_prefix(self):
        q, k, v = [torch.randn(9, 65, device=DEVICE) for _ in range(3)]
        expected = FORWARD(q, k, v, is_causal=True)
        k[4:] = torch.randn_like(k[4:]) * 10
        v[4:] = torch.randn_like(v[4:]) * 10
        actual = FORWARD(q, k, v, is_causal=True)
        torch.testing.assert_close(actual[:4], expected[:4], rtol=0, atol=0)

    def test_changing_softmax_maximum(self):
        scores = torch.tensor([-30, 2, 2, -70, 40, 1, 60], device=DEVICE, dtype=torch.float32)
        q = torch.zeros(7, 33, device=DEVICE)
        q[:, 0] = 1
        k = torch.zeros_like(q)
        k[:, 0] = scores * 33**0.5
        v = torch.randn_like(q)
        for causal in (False, True):
            verify_against_cpu_fp64(FORWARD, q, k, v, is_causal=causal)

    def test_key_tiles_causal_diagonals_and_query_tails(self):
        for seq_len in (8, 9, 31, 32, 33, 63, 64, 65):
            for head_dim in (3, 33, 64, 97, 128):
                inputs = [torch.randn(seq_len, head_dim, device=DEVICE) for _ in range(3)]
                for causal in (False, True):
                    with self.subTest(shape=(seq_len, head_dim), causal=causal):
                        verify_against_cpu_fp64(FORWARD, *inputs, is_causal=causal)

    def test_softmax_maximum_changes_across_tiles(self):
        q = torch.zeros(65, 64, device=DEVICE)
        q[:, 0] = 1
        k = torch.zeros_like(q)
        scores = torch.full((65,), -70.0, device=DEVICE)
        scores[[0, 31, 32, 63, 64]] = torch.tensor(
            [-30, 2, 40, 40, 60], device=DEVICE, dtype=scores.dtype,
        )
        k[:, 0] = scores * 64**0.5
        v = torch.randn_like(q)
        for causal in (False, True):
            verify_against_cpu_fp64(FORWARD, q, k, v, is_causal=causal)

    def test_cancellation_across_accumulator_groups(self):
        for head_dim in (3, 4, 5, 31, 33, 63, 64, 65, 127, 128):
            q = torch.randn(33, head_dim, device=DEVICE)
            k = torch.randn_like(q)
            signs = 1 - 2 * (torch.arange(head_dim, device=DEVICE) % 2)
            q *= signs
            v = torch.randn_like(q)
            v[1::2] = -v[::2][:16]
            for causal in (False, True):
                with self.subTest(head_dim=head_dim, causal=causal):
                    verify_against_cpu_fp64(FORWARD, q, k, v, is_causal=causal)

    def test_subnormal_probabilities_preserve_large_value_contributions(self):
        for head_dim in (33, 64):
            for score_gap in (86.0, 88.0, 90.0, 96.0):
                q = torch.zeros(65, head_dim, device=DEVICE)
                q[:, 0] = 1.0
                k = torch.zeros_like(q)
                k[:64, 0] = -score_gap * head_dim**0.5
                v = torch.zeros_like(q)
                v[:64] = 1e36
                for causal in (False, True):
                    with self.subTest(head_dim=head_dim, score_gap=score_gap, causal=causal):
                        verify_against_cpu_fp64(FORWARD, q, k, v, is_causal=causal)

    def test_contiguous_inputs_with_unaligned_storage_offset(self):
        for head_dim in (4, 32, 64, 128):
            for offset in (1, 2, 3):
                inputs = [
                    torch.randn(65 * head_dim + offset, device=DEVICE)[offset:].view(65, head_dim)
                    for _ in range(3)
                ]
                self.assertTrue(all(tensor.is_contiguous() for tensor in inputs))
                for causal in (False, True):
                    with self.subTest(head_dim=head_dim, offset=offset, causal=causal):
                        verify_against_cpu_fp64(FORWARD, *inputs, is_causal=causal)

    def test_rejects_noncontiguous_inputs(self):
        q, k, v = [
            torch.randn(64, 17, device=DEVICE).T
            for _ in range(3)
        ]
        self.assertFalse(q.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_cuda_fa2_support(q, k, v)

    def test_rejects_unsupported_head_dimension(self):
        q, k, v = [
            torch.randn(7, 129, device=DEVICE)
            for _ in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "d <= 128"):
            validate_cuda_fa2_support(q, k, v)


class CudaFa2LongSequenceTests(unittest.TestCase):
    report = None

    def setUp(self):
        if not RUN_LONG_SEQUENCE or DEVICE.type != "cuda":
            self.skipTest("sampled long-sequence validation requires --device cuda --long-sequence")

    def test_sampled_rows_against_cpu_fp64(self):
        torch.manual_seed(0)
        inputs = [torch.randn(16384, 64, device=DEVICE) for _ in range(3)]
        cpu_inputs = [tensor.cpu().double() for tensor in inputs]
        records = []
        for causal in (False, True):
            output = FORWARD(*inputs, is_causal=causal).cpu().double()
            for qi in (0, 1, 3, 4, 7, 8, 31, 32, 8191, 16383):
                with self.subTest(causal=causal, query_index=qi):
                    expected = reference_row(*cpu_inputs, qi, causal)
                    torch.testing.assert_close(
                        output[qi], expected, rtol=REFERENCE_TOLERANCE, atol=REFERENCE_TOLERANCE,
                    )
                    records.append({
                        "causal": causal,
                        "query_index": qi,
                        "max_abs_error": (output[qi] - expected).abs().max().item(),
                    })
        type(self).report = {
            "implementation": IMPLEMENTATION,
            **implementation_metadata(IMPLEMENTATION),
            "shape": [16384, 64],
            "seed": 0,
            "cpu_threads": 1,
            "rtol": REFERENCE_TOLERANCE,
            "atol": REFERENCE_TOLERANCE,
            "verified_all_rows": False,
            "reference": "CPU FP64 row dot/softmax/PV; no full attention matrix",
            "rows": records,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--impl", choices=("cuda_fa2", "cuda_fa2_v1", "cuda_fa2_v2", "cuda_fa2_v3", "cuda_fa2_v4"),
        default="cuda_fa2",
    )
    parser.add_argument("--long-sequence", action="store_true", help="Also check sampled rows at S=16384, d=64")
    parser.add_argument("--output-json", type=Path, help="Save the optional long-sequence verification result")
    args = parser.parse_args()
    RUN_LONG_SEQUENCE = args.long_sequence
    if args.long_sequence and args.device != "cuda":
        parser.error("--long-sequence requires --device cuda")
    if args.output_json and not args.long_sequence:
        parser.error("--output-json requires --long-sequence")
    IMPLEMENTATION = args.impl
    FORWARD = get_attention_forward(IMPLEMENTATION)
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the environment and GPU permissions")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    classes = [CudaFa2ContractTests, CudaFa2NumericalTests]
    if args.long_sequence:
        classes.append(CudaFa2LongSequenceTests)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful() and args.output_json:
        serialized = json.dumps(CudaFa2LongSequenceTests.report, indent=2, allow_nan=False)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
