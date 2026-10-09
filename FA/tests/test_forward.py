"""Run from FA with: python -m tests.test_forward --device cpu|cuda."""

import argparse
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.forward import attention_forward


DEVICE = torch.device("cpu")


class AttentionForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def assert_matches_reference(self, q, k, v, is_causal):
        # FP64 CPU math SDPA is only an oracle; forward.py never calls SDPA.
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(
                q.cpu().double(), k.cpu().double(), v.cpu().double(),
                dropout_p=0.0, is_causal=is_causal,
            )
        actual = attention_forward(q, k, v, is_causal=is_causal)
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(actual.shape, q.shape)
        self.assertEqual(actual.dtype, q.dtype)
        self.assertEqual(actual.device, q.device)

    def test_shapes_against_sdpa(self):
        cases = [(1, 1), (7, 3), (19, 16), (65, 32), (257, 64)]
        for seq_len, d in cases:
            for causal in (False, True):
                with self.subTest(seq_len=seq_len, d=d, causal=causal):
                    q = torch.randn(seq_len, d, dtype=torch.float32, device=DEVICE)
                    k = torch.randn_like(q)
                    v = torch.randn_like(k)
                    self.assert_matches_reference(q, k, v, causal)

    def test_noncontiguous_inputs(self):
        q = torch.randn(16, 23, device=DEVICE).T
        k = torch.randn(16, 23, device=DEVICE).T
        v = torch.randn_like(k)
        self.assertFalse(q.is_contiguous())
        for causal in (False, True):
            with self.subTest(causal=causal):
                self.assert_matches_reference(q, k, v, causal)

    def test_uniform_scores_and_causal_prefix_average(self):
        for seq_len in (7, 13):
            q = torch.zeros(seq_len, 2, device=DEVICE)
            k = torch.zeros_like(q)
            v = torch.arange(seq_len * 2, dtype=torch.float32, device=DEVICE).reshape(seq_len, 2)
            expected = torch.stack([v[:i + 1].mean(dim=0) for i in range(seq_len)])
            torch.testing.assert_close(attention_forward(q, k, v, is_causal=True), expected)
            torch.testing.assert_close(attention_forward(q, k, v), v.mean(dim=0).expand_as(q))

    def test_future_tokens_cannot_change_earlier_outputs(self):
        q = torch.randn(9, 4, device=DEVICE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        before = attention_forward(q, k, v, is_causal=True)
        k[5:] += 100
        v[5:] -= 100
        after = attention_forward(q, k, v, is_causal=True)
        torch.testing.assert_close(before[:5], after[:5])

    def test_large_finite_logits_are_stable(self):
        q = torch.full((5, 7), 100.0, device=DEVICE)
        k = torch.full_like(q, 100.0)
        v = torch.randn_like(q)
        actual = attention_forward(q, k, v)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, v.mean(dim=0).expand_as(q))

    def test_forward_only_and_inputs_unchanged(self):
        inputs = [torch.randn(5, 4, device=DEVICE, requires_grad=True) for _ in range(3)]
        copies = [tensor.detach().clone() for tensor in inputs]
        output = attention_forward(*inputs, is_causal=True)
        self.assertFalse(output.requires_grad)
        for tensor, copy in zip(inputs, copies):
            torch.testing.assert_close(tensor, copy)

    def test_invalid_inputs(self):
        q = torch.randn(3, 4, device=DEVICE)
        cases = [
            (q.flatten(), q, q),
            (q, q[:, :3], q[:, :3]),
            (q, q, q[:2]),
            (q, q.double(), q.double()),
            (q[:0], q, q),
            (q[:, :0], q[:, :0], q[:, :0]),
            (q, q[:2], q[:2]),
            (q[None], q[None], q[None]),
            (q[None, None], q[None, None], q[None, None]),
        ]
        for inputs in cases:
            with self.subTest(shapes=[tuple(t.shape) for t in inputs]):
                with self.assertRaises(ValueError):
                    attention_forward(*inputs)
        for dtype in (torch.int32, torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(rejected_dtype=dtype):
                with self.assertRaisesRegex(TypeError, "float32"):
                    attention_forward(q.to(dtype), q.to(dtype), q.to(dtype))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the environment and GPU device permissions")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AttentionForwardTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
