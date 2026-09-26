from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cs336_basics.nn_utils import cross_entropy


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def main() -> None:
    torch.manual_seed(0)
    model = ToyModel(in_features=16, out_features=8)
    inputs = torch.randn(4, 16)
    targets = torch.randint(0, 8, (4,))
    output_dtypes: dict[str, torch.dtype] = {}

    def capture_output_dtype(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            output_dtypes[name] = output.dtype

        return hook

    handles = [
        model.fc1.register_forward_hook(capture_output_dtype("fc1")),
        model.ln.register_forward_hook(capture_output_dtype("layer_norm")),
        model.fc2.register_forward_hook(capture_output_dtype("fc2")),
    ]
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            parameter_dtype_inside = next(model.parameters()).dtype
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            project_loss = cross_entropy(logits, targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None})
    print(f"torch_version={torch.__version__}")
    print("experiment_device=cpu")
    print("autocast_dtype=torch.bfloat16")
    print(f"parameter_dtype_inside={parameter_dtype_inside}")
    print(f"fc1_output_dtype={output_dtypes['fc1']}")
    print(f"layer_norm_output_dtype={output_dtypes['layer_norm']}")
    print(f"logits_dtype={logits.dtype}")
    print(f"builtin_cross_entropy_dtype={loss.dtype}")
    print(f"project_cross_entropy_dtype={project_loss.dtype}")
    print(f"gradient_dtypes={gradient_dtypes}")


if __name__ == "__main__":
    main()
