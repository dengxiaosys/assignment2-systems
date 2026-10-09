"""Custom C++/CUDA FA2-style forward tests."""

import argparse
import unittest

import torch

from src.cuda_fa2_forward import (
    cuda_fa2_attention_forward,
    validate_cuda_fa2_support,
)
from src.numerical_verification import verify_against_cpu_fp64


DEVICE = torch.device("cpu")


class CudaFa2ContractTests(unittest.TestCase):
    def test_rejects_cpu_before_loading_extension(self):
        q = torch.randn(7, 16)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            validate_cuda_fa2_support(q, q, q)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            cuda_fa2_attention_forward(q, q, q)


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
                        cuda_fa2_attention_forward,
                        *inputs,
                        is_causal=causal,
                    )
                    output = cuda_fa2_attention_forward(
                        *inputs,
                        is_causal=causal,
                    )
                    self.assertEqual(output.shape, inputs[0].shape)
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertEqual(output.device, inputs[0].device)
                    self.assertFalse(output.requires_grad)
                    for tensor, copy in zip(inputs, copies):
                        torch.testing.assert_close(tensor, copy)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the environment and GPU permissions")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
