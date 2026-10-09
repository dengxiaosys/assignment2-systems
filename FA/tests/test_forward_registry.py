"""Named attention forward selection tests."""

import unittest
from unittest.mock import patch

import torch

from src.forward_registry import (
    get_attention_forward,
    validate_implementation,
)
from src.memory_efficient_forward import memory_efficient_attention_forward
from src.native_forward import native_attention_forward


class ForwardRegistryTests(unittest.TestCase):
    def test_resolves_named_implementations(self):
        self.assertIs(get_attention_forward("native"), native_attention_forward)
        self.assertIs(
            get_attention_forward("efficient"),
            memory_efficient_attention_forward,
        )

    def test_rejects_unknown_implementation(self):
        q = torch.randn(7, 16)
        for name in ("auto", "flash"):
            with self.subTest(implementation=name):
                with self.assertRaisesRegex(ValueError, "Unknown implementation"):
                    get_attention_forward(name)
                with self.assertRaisesRegex(ValueError, "Unknown implementation"):
                    validate_implementation(name, q, q, q)

    def test_delegates_memory_efficient_support_check(self):
        q = torch.randn(7, 16)
        with patch(
            "src.forward_registry.validate_memory_efficient_support"
        ) as validate_support:
            validate_implementation("efficient", q, q, q, is_causal=True)
        validate_support.assert_called_once_with(
            q,
            q,
            q,
            is_causal=True,
        )

