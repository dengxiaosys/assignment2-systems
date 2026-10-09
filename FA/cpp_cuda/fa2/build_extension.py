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
PROJECT_ROOT = SOURCE_DIR.parents[1]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _configure_cuda_home()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup(
        name="cs336-cuda-fa2",
        packages=["src"],
        package_dir={"src": "src"},
        ext_modules=[
            CUDAExtension(
                name="src.cuda_fa2_extension",
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
