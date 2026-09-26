from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from cs336_basics.model import TransformerBlock, TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def _dtype_counts(tensors) -> dict[str, int]:
    counts = Counter(str(tensor.dtype) for tensor in tensors)
    return dict(sorted(counts.items()))


def _autocast_context(dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cpu", dtype=dtype)


def inspect_training_step(label: str, autocast_dtype: torch.dtype | None) -> None:
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=128,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
        device="cpu",
        dtype=torch.float32,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    input_ids = torch.randint(0, 128, (2, 16))
    targets = torch.randint(0, 128, (2, 16))
    activations: dict[str, Tensor] = {}
    saved_counts: Counter[str] = Counter()
    saved_bytes: Counter[str] = Counter()

    def capture_activation(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            output.retain_grad()
            activations[name] = output

        return hook

    def pack_saved_tensor(tensor: Tensor) -> Tensor:
        dtype_name = str(tensor.dtype)
        saved_counts[dtype_name] += 1
        saved_bytes[dtype_name] += tensor.numel() * tensor.element_size()
        return tensor

    def unpack_saved_tensor(tensor: Tensor) -> Tensor:
        return tensor

    first_layer = model.layers[0]
    assert isinstance(first_layer, TransformerBlock)
    handles = [
        model.token_embeddings.register_forward_hook(capture_activation("embedding_output")),
        first_layer.ln1.register_forward_hook(capture_activation("first_rmsnorm_output")),
        first_layer.attn.q_proj.register_forward_hook(capture_activation("first_q_projection")),
        model.lm_head.register_forward_hook(capture_activation("logits")),
    ]

    parameter_tensors = list(model.parameters())
    first_parameter_before = parameter_tensors[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    print(f"run={label} phase=before_forward parameter_dtypes={json.dumps(_dtype_counts(parameter_tensors), sort_keys=True)}")
    print(f"run={label} phase=before_forward gradients_are_none={all(parameter.grad is None for parameter in parameter_tensors)}")
    print(f"run={label} phase=before_forward optimizer_state_entries={len(optimizer.state)}")

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_saved_tensor, unpack_saved_tensor):
            with _autocast_context(autocast_dtype):
                logits = model(input_ids)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            print(f"run={label} phase=after_forward loss_dtype={loss.dtype}")
            for name, activation in activations.items():
                print(f"run={label} phase=after_forward activation={name} dtype={activation.dtype}")
            print(f"run={label} phase=after_forward saved_tensor_counts={json.dumps(dict(sorted(saved_counts.items())))}")
            print(f"run={label} phase=after_forward saved_tensor_bytes={json.dumps(dict(sorted(saved_bytes.items())))}")

            loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    parameter_gradients = [parameter.grad for parameter in parameter_tensors if parameter.grad is not None]
    print(f"run={label} phase=after_backward parameter_gradient_dtypes={json.dumps(_dtype_counts(parameter_gradients), sort_keys=True)}")
    for name, activation in activations.items():
        gradient_dtype = None if activation.grad is None else str(activation.grad.dtype)
        print(f"run={label} phase=after_backward activation={name} gradient_dtype={gradient_dtype}")

    optimizer.step()
    state_tensors = [value for state in optimizer.state.values() for value in state.values() if isinstance(value, Tensor)]
    parameter_update = (parameter_tensors[0].detach() - first_parameter_before).abs().max().item()
    print(f"run={label} phase=after_optimizer optimizer_state_entries={len(optimizer.state)}")
    print(f"run={label} phase=after_optimizer optimizer_state_tensor_dtypes={json.dumps(_dtype_counts(state_tensors), sort_keys=True)}")
    print(f"run={label} phase=after_optimizer parameter_dtypes={json.dumps(_dtype_counts(parameter_tensors), sort_keys=True)}")
    print(f"run={label} phase=after_optimizer first_parameter_max_abs_update={parameter_update:.9g}")


def main() -> None:
    print(f"torch_version={torch.__version__}")
    inspect_training_step("fp32", autocast_dtype=None)
    inspect_training_step("cpu_bfloat16_autocast", autocast_dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
