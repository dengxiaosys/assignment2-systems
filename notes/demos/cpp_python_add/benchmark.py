from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from cpp_add_demo import add as cpp_add


def python_add(a: int, b: int) -> int:
    return a + b


def measure(function: Callable[[int, int], int], iterations: int) -> tuple[float, int]:
    checksum = 0
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        checksum += function(1, 2)
    elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1e9
    return elapsed_seconds, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000_000)
    args = parser.parse_args()

    for _ in range(10_000):
        python_add(1, 2)
        cpp_add(1, 2)

    python_seconds, python_checksum = measure(python_add, args.iterations)
    cpp_seconds, cpp_checksum = measure(cpp_add, args.iterations)

    print(f"iterations={args.iterations}")
    print(f"python_seconds={python_seconds:.6f}")
    print(f"cpp_seconds={cpp_seconds:.6f}")
    print(f"cpp_over_python_ratio={cpp_seconds / python_seconds:.3f}")
    print(f"python_checksum={python_checksum}")
    print(f"cpp_checksum={cpp_checksum}")


if __name__ == "__main__":
    main()
