"""Python wrapper for the prebuilt C++/CUDA attention forward."""

from types import ModuleType

import torch

from .input_validation import validate_attention_inputs


MAX_HEAD_DIM = 128
_BUILD_COMMAND = (
    "python cpp_cuda/fa2/build_extension.py build_ext --inplace"
)


def import_cuda_fa2_extension() -> ModuleType:
    """Import the generated src/cuda_fa2_extension shared library."""
    try:
        import src.cuda_fa2_extension as extension
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "cuda_fa2 extension is not built or cannot be loaded. "
            f"Run `{_BUILD_COMMAND}` from the FA directory."
        ) from error
    return extension


def validate_cuda_fa2_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> None:
    """Validate the supported subset and ensure the extension is importable."""
    del is_causal
    validate_attention_inputs(q, k, v)
    if q.device.type != "cuda":
        raise RuntimeError("cuda_fa2 requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("cuda_fa2 requires contiguous Q/K/V")
    if q.shape[1] > MAX_HEAD_DIM:
        raise ValueError(f"cuda_fa2 requires d <= {MAX_HEAD_DIM}")
    capability = torch.cuda.get_device_capability(q.device)
    if capability < (6, 0):
        raise RuntimeError(
            f"cuda_fa2 requires compute capability >= 6.0, got {capability}"
        )
    import_cuda_fa2_extension()


@torch.no_grad()
def cuda_fa2_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run the custom online-softmax CUDA baseline."""
    validate_cuda_fa2_support(q, k, v, is_causal=is_causal)
    extension = import_cuda_fa2_extension()
    return extension.forward(q, k, v, is_causal)
