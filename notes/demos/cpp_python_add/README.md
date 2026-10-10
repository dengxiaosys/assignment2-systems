# C++ `a + b` Python Extension Demo

This directory is the runnable companion to
[`07_01_cpp_function_python_extension_textbook.md`](../../07_01_cpp_function_python_extension_textbook.md).

## Build and run

From the Assignment 2 root:

```bash
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  uv sync --project notes/demos/cpp_python_add --group test

uv run --project notes/demos/cpp_python_add \
  python -c 'from cpp_add_demo import add; print(f"result={add(2, 3)}")'

uv run --project notes/demos/cpp_python_add \
  pytest -q notes/demos/cpp_python_add/tests
```

## Inspect the extension

```bash
uv run --project notes/demos/cpp_python_add \
  python -c '
import cpp_add_demo._cpp_add as module
print(f"extension_path={module.__file__}")
'
```

On Linux, inspect the generated shared object with:

```bash
extension_path="$(
  uv run --project notes/demos/cpp_python_add \
    python -c 'import cpp_add_demo._cpp_add as module; print(module.__file__)'
)"

file "$extension_path"
ldd "$extension_path"
nm -D "$extension_path" | grep PyInit
```

## Benchmark the boundary

```bash
uv run --project notes/demos/cpp_python_add \
  python notes/demos/cpp_python_add/benchmark.py \
  --iterations 1000000
```

The native scalar function is expected to be slower than Python's built-in integer
addition because every call crosses the Python/C++ boundary. Native code becomes
useful when one boundary crossing performs enough work to amortize conversion and
dispatch overhead.
