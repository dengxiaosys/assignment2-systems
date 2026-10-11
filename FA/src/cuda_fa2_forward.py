"""Load, identify and run the prebuilt C++/CUDA attention versions."""

import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import torch

from .input_validation import validate_attention_inputs


MAX_HEAD_DIM = 128


def import_cuda_fa2_extension(version: str = "v0") -> ModuleType:
    """Import the selected extension without rebuilding or replacing another version."""
    if version == "v0":
        module_name = "src.cuda_fa2_extension"
    elif version in ("v1", "v2", "v3", "v4"):
        module_name = f"src.cuda_fa2_{version}_extension"
    else:
        raise ValueError(f"Unknown CUDA FA2 version {version!r}")
    try:
        extension = importlib.import_module(module_name)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"{module_name} is not built or cannot be loaded. "
            f"Run `python cpp_cuda/fa2/build_extension.py --version {version} "
            "build_ext --inplace` from the FA directory."
        ) from error
    return extension


def validate_cuda_fa2_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    version: str = "v0",
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
    import_cuda_fa2_extension(version)


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


@torch.no_grad()
def cuda_fa2_v1_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run four independent query warps per CTA with register shuffle communication."""
    validate_cuda_fa2_support(q, k, v, is_causal=is_causal, version="v1")
    extension = import_cuda_fa2_extension("v1")
    return extension.forward(q, k, v, is_causal)


@torch.no_grad()
def cuda_fa2_v2_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run eight query warps sharing 32-key tiles with tile-level online softmax."""
    validate_cuda_fa2_support(q, k, v, is_causal=is_causal, version="v2")
    extension = import_cuda_fa2_extension("v2")
    return extension.forward(q, k, v, is_causal)


@torch.no_grad()
def cuda_fa2_v3_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run SIMT register matrix tiles with shared storage reused between QK and PV."""
    validate_cuda_fa2_support(q, k, v, is_causal=is_causal, version="v3")
    extension = import_cuda_fa2_extension("v3")
    return extension.forward(q, k, v, is_causal)


@torch.no_grad()
def cuda_fa2_v4_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run 64-key SIMT matrix tiles with staged prefetch and aligned vector loads."""
    validate_cuda_fa2_support(q, k, v, is_causal=is_causal, version="v4")
    extension = import_cuda_fa2_extension("v4")
    return extension.forward(q, k, v, is_causal)


def implementation_metadata(implementation: str) -> dict:
    """Record the loaded CUDA extension identity outside benchmark timing."""
    if implementation not in ("cuda_fa2", "cuda_fa2_v1", "cuda_fa2_v2", "cuda_fa2_v3", "cuda_fa2_v4"):
        return {}
    version = "v0" if implementation == "cuda_fa2" else implementation.removeprefix("cuda_fa2_")
    extension = import_cuda_fa2_extension(version)
    path = Path(extension.__file__).resolve()
    return {
        "cuda_fa2_version": version,
        "extension_module": extension.__name__,
        "extension_path": str(path),
        "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
