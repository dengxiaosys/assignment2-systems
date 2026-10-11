"""Build a selected FA2 version with one shared binding and setuptools entry."""

import argparse
import os
import shutil
import sys
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", choices=("v0", "v1", "v2"), default="v1")
    args, setup_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *setup_args]
    module_name = (
        "src.cuda_fa2_extension" if args.version == "v0"
        else f"src.cuda_fa2_{args.version}_extension"
    )

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MAX_JOBS", "1")
    _configure_cuda_home()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup(
        name=f"cs336-cuda-fa2-{args.version}",
        packages=["src"],
        package_dir={"src": "src"},
        ext_modules=[
            CUDAExtension(
                name=module_name,
                sources=[
                    str(SOURCE_DIR / "common" / "binding.cpp"),
                    str(SOURCE_DIR / args.version / "fa2_forward.cu"),
                ],
                extra_compile_args={
                    "cxx": ["-O2"],
                    "nvcc": ["-O2", "-lineinfo", "--ptxas-options=-v"],
                },
            )
        ],
        cmdclass={
            "build_ext": BuildExtension.with_options(use_ninja=False),
        },
    )


if __name__ == "__main__":
    main()
