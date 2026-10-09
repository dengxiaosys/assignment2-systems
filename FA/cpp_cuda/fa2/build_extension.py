"""Build the custom CUDA attention extension without Ninja."""

import os
import shutil
from pathlib import Path


def _configure_cuda_home() -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return
    cuda_home = Path(nvcc).resolve().parents[1]
    if (cuda_home / "include" / "cuda_runtime.h").is_file():
        os.environ["CUDA_HOME"] = str(cuda_home)


SOURCE_DIR = Path(__file__).resolve().parent


def main() -> None:
    _configure_cuda_home()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup(
        name="cs336-cuda-fa2",
        packages=["src"],
        ext_modules=[
            CUDAExtension(
                name="src._cuda_fa2",
                sources=[
                    str(SOURCE_DIR / "binding.cpp"),
                    str(SOURCE_DIR / "fa2_forward.cu"),
                ],
                extra_compile_args={
                    "cxx": ["-O2"],
                    "nvcc": ["-O2", "-lineinfo"],
                },
            )
        ],
        cmdclass={
            "build_ext": BuildExtension.with_options(use_ninja=False),
        },
    )


if __name__ == "__main__":
    main()
