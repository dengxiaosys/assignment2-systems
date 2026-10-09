"""Memory-efficient forward tests; --device cuda checks real fused outputs."""

import argparse
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.memory_efficient_forward import (
    memory_efficient_attention_forward,
    validate_memory_efficient_support,
)
from src.numerical_verification import verify_against_cpu_fp64


DEVICE = torch.device("cpu")


def enabled_sdpa_backends():
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


class MemoryEfficientForwardContractTests(unittest.TestCase):
    def test_efficient_rejects_cpu(self):
        q = torch.randn(7, 16)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            validate_memory_efficient_support(q, q, q)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            memory_efficient_attention_forward(q, q, q)

    def test_forced_backend_views_and_state_restoration(self):
        # Mock only the CUDA boundary and operator: verifies dispatch, not GPU math.
        q, k, v = [torch.randn(7, 64, dtype=torch.float32, requires_grad=True) for _ in range(3)]
        all_backends = [
            SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION,
        ]
        for fail in (False, True):
            with self.subTest(failure=fail):
                def fake_sdpa(q4, k4, v4, *, dropout_p, is_causal):
                    self.assertEqual(enabled_sdpa_backends(), (False, True, False, False))
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertEqual(dropout_p, 0.0)
                    self.assertTrue(is_causal)
                    for view, original in zip((q4, k4, v4), (q, k, v)):
                        self.assertEqual(view.shape, (1, 1, 7, 64))
                        self.assertEqual(view.dtype, torch.float32)
                        self.assertEqual(view.data_ptr(), original.data_ptr())
                    if fail:
                        raise RuntimeError("unsupported kernel")
                    return q4 + k4 + v4

                with sdpa_kernel(all_backends):
                    before = enabled_sdpa_backends()
                    with (
                        patch("src.memory_efficient_forward._require_cuda"),
                        patch(
                            "src.memory_efficient_forward.F.scaled_dot_product_attention",
                            side_effect=fake_sdpa,
                        ) as sdpa,
                    ):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "unsupported kernel"):
                                memory_efficient_attention_forward(
                                    q,
                                    k,
                                    v,
                                    is_causal=True,
                                )
                        else:
                            output = memory_efficient_attention_forward(
                                q,
                                k,
                                v,
                                is_causal=True,
                            )
                            self.assertEqual(output.shape, q.shape)
                            self.assertEqual(output.dtype, torch.float32)
                            self.assertFalse(output.requires_grad)
                        sdpa.assert_called_once()
                    self.assertEqual(enabled_sdpa_backends(), before)


class MemoryEfficientCudaNumericalTests(unittest.TestCase):
    def setUp(self):
        if DEVICE.type != "cuda":
            self.skipTest("real GPU numerical validation requires --device cuda")
        torch.manual_seed(0)

    def test_matches_cpu_fp64_reference(self):
        probe = torch.randn(65, 64, device=DEVICE, dtype=torch.float32)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                validate_memory_efficient_support(probe, probe, probe)
        except RuntimeError as error:
            self.skipTest(str(error))
        for seq_len in (7, 65, 257):
            for causal in (False, True):
                with self.subTest(seq_len=seq_len, causal=causal):
                    inputs = [
                        torch.randn(seq_len, 64, device=DEVICE, dtype=torch.float32, requires_grad=True)
                        for _ in range(3)
                    ]
                    copies = [tensor.detach().clone() for tensor in inputs]
                    validate_memory_efficient_support(
                        *inputs,
                        is_causal=causal,
                    )
                    verify_against_cpu_fp64(
                        memory_efficient_attention_forward,
                        *inputs,
                        is_causal=causal,
                    )
                    output = memory_efficient_attention_forward(
                        *inputs,
                        is_causal=causal,
                    )
                    self.assertEqual(output.shape, inputs[0].shape)
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertEqual(output.device, inputs[0].device)
                    self.assertFalse(output.requires_grad)
                    for tensor, copy in zip(inputs, copies):
                        torch.testing.assert_close(tensor, copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the environment and GPU device permissions")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
