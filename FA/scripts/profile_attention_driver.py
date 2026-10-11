"""Profile the prebuilt FA kernel using CUDA Driver API and Nsight Compute 2019.5.

This entry point uses only the Python standard library and does not load PyTorch.
Q and K are zero, so attention is a uniform average of deterministic V rows.
It is a profiler compatibility/learning fixture, not the random-input benchmark.
"""

import argparse
import array
import ctypes as ct
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUOBJDUMP = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"


class CudaDriver:
    """Bind the small, stable Driver API subset needed to launch a cubin."""

    def __init__(self) -> None:
        self.library = ct.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuCtxCreate_v2": [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
            "cuCtxDestroy_v2": [ct.c_void_p],
            "cuModuleLoad": [ct.POINTER(ct.c_void_p), ct.c_char_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemcpyHtoD_v2": [ct.c_uint64, ct.c_void_p, ct.c_size_t],
            "cuMemcpyDtoH_v2": [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            "cuLaunchKernel": [
                ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p,
                ct.POINTER(ct.c_void_p), ct.c_void_p,
            ],
            "cuCtxSynchronize": [],
            "cuProfilerStart": [],
            "cuProfilerStop": [],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, parameters in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = parameters
            function.restype = ct.c_int

    def call(self, name: str, *arguments) -> None:
        result = getattr(self.library, name)(*arguments)
        if result:
            message = ct.c_char_p()
            self.library.cuGetErrorString(result, ct.byref(message))
            raise RuntimeError(f"{name}: CUDA error {result}, {message.value!r}")


def extract_kernel(directory: Path) -> tuple[Path, bytes]:
    """Extract the actual sm_61 binary and discover its mangled kernel name."""
    extensions = list((PROJECT_ROOT / "src").glob("cuda_fa2_extension*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one prebuilt FA extension, found {extensions}")
    subprocess.run(
        [CUOBJDUMP, "--extract-elf", "all", str(extensions[0])],
        cwd=directory, check=True, capture_output=True, text=True,
    )
    cubins = list(directory.glob("*.sm_61.cubin"))
    if len(cubins) != 1:
        raise RuntimeError(f"Expected one sm_61 cubin, found {cubins}")
    resources = subprocess.check_output(
        [CUOBJDUMP, "--dump-resource-usage", str(cubins[0])], text=True,
    )
    symbol = re.search(r"Function\s+(\S*fa2_forward_fp32_kernel\S*):", resources)
    if symbol is None:
        raise RuntimeError("Could not find fa2_forward_fp32_kernel in the cubin")
    return cubins[0], symbol.group(1).encode()


def uniform_attention_error(
    output: array.array, values: array.array, seq_len: int, head_dim: int, causal: bool,
) -> float:
    """Check the uniform-attention fixture in O(Sd) time and O(d) extra space."""
    totals = [0.0] * head_dim
    if not causal:
        for index, value in enumerate(values):
            totals[index % head_dim] += value
    error = 0.0
    for row in range(seq_len):
        for dim in range(head_dim):
            index = row * head_dim + dim
            if causal:
                totals[dim] += values[index]
            expected = totals[dim] / (row + 1 if causal else seq_len)
            error = max(error, abs(output[index] - expected))
    return error


def profile(cubin: Path, symbol: bytes, args: argparse.Namespace) -> None:
    driver = CudaDriver()
    driver.call("cuInit", 0)
    device, context, module, function = ct.c_int(), ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    driver.call("cuDeviceGet", ct.byref(device), 0)
    driver.call("cuCtxCreate_v2", ct.byref(context), 0, device)
    try:
        driver.call("cuModuleLoad", ct.byref(module), str(cubin).encode())
        driver.call("cuModuleGetFunction", ct.byref(function), module, symbol)
        count = args.seq_len * args.head_dim
        zeros = array.array("f", [0.0]) * count
        values = array.array("f", ((index % 17 - 8) / 8 for index in range(count)))
        output = array.array("f", [0.0]) * count
        size = count * zeros.itemsize
        allocations = [ct.c_uint64() for _ in range(4)]
        for pointer in allocations:
            driver.call("cuMemAlloc_v2", ct.byref(pointer), size)
        for pointer, host in zip(allocations[:3], (zeros, zeros, values)):
            driver.call("cuMemcpyHtoD_v2", pointer, host.buffer_info()[0], size)
        arguments = allocations + [
            ct.c_int(args.seq_len), ct.c_int(args.head_dim),
            ct.c_float(1 / math.sqrt(args.head_dim)), ct.c_bool(args.causal),
        ]
        parameters = (ct.c_void_p * len(arguments))(
            *(ct.addressof(argument) for argument in arguments)
        )

        def launch() -> None:
            driver.call(
                "cuLaunchKernel", function, args.seq_len, 1, 1, 128, 1, 1, 0,
                None, parameters, None,
            )

        for _ in range(args.warmup):
            launch()
        driver.call("cuCtxSynchronize")
        driver.call("cuProfilerStart")
        try:
            launch()
            driver.call("cuCtxSynchronize")
        finally:
            driver.call("cuProfilerStop")
        driver.call("cuMemcpyDtoH_v2", output.buffer_info()[0], allocations[3], size)
        error = uniform_attention_error(
            output, values, args.seq_len, args.head_dim, args.causal,
        )
        if not math.isfinite(error) or error > 1e-5 or any(not math.isfinite(x) for x in output):
            raise RuntimeError(f"Uniform attention verification failed: {error}")
        print(json.dumps({
            "entry_point": "CUDA Driver API",
            "input_fixture": "Q=K=0, deterministic V; uniform attention",
            "shape": [args.seq_len, args.head_dim],
            "causal": args.causal,
            "warmup": args.warmup,
            "captured_forward_calls": 1,
            "max_abs_error": error,
        }, indent=2), flush=True)
        for pointer in allocations:
            driver.call("cuMemFree_v2", pointer)
        driver.call("cuModuleUnload", module)
    finally:
        # Destroying the dedicated context also releases allocations after errors.
        driver.call("cuCtxDestroy_v2", context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.seq_len <= 0 or not 1 <= args.head_dim <= 128 or args.warmup < 0:
        parser.error("Require S > 0, 1 <= d <= 128 and warmup >= 0")
    if args.seq_len * args.head_dim > 2**31 - 1:
        parser.error("Kernel uses 32-bit indexing; require S*d <= INT_MAX")
    with tempfile.TemporaryDirectory(prefix="fa2-cubin-") as temporary:
        cubin, symbol = extract_kernel(Path(temporary))
        profile(cubin, symbol, args)


if __name__ == "__main__":
    main()
