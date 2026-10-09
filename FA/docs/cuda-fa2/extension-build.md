# C++/CUDA Extension 构建脚本说明

## 1. 脚本职责

[build_extension.py](../../cpp_cuda/fa2/build_extension.py) 负责将以下源码编译为可被 Python 导入的动态扩展：

- [binding.cpp](../../cpp_cuda/fa2/binding.cpp#L1-L14)：定义 Pybind11 模块并暴露 `forward`。
- [fa2_forward.cu](../../cpp_cuda/fa2/fa2_forward.cu#L18-L143)：实现 CUDA Kernel 与 C++ Launch Wrapper。

最终产物为：

```text
src/cuda_fa2_extension.cpython-<python-version>-<platform>.so
```

Python 侧通过以下语句导入：

```python
import src.cuda_fa2_extension as extension
```

该脚本只负责构建，不负责运行正确性测试、Benchmark 或 Profiling。

## 2. 为什么使用显式构建

当前实现采用显式的 Ahead-of-Time Build：

```bash
python cpp_cuda/fa2/build_extension.py build_ext --inplace
```

没有在第一次 Forward 时调用 `torch.utils.cpp_extension.load()` 隐式编译，原因如下：

- 编译时间不会混入 Benchmark 或 Profiling 流程。
- 构建错误和运行错误能够明确区分。
- 源码修改后是否重新构建由开发者显式控制。
- 不需要在运行时处理 Ninja、编译器 PATH 或临时缓存目录。
- 生成的 `.so` 路径固定，便于理解 Python 与 C++/CUDA 之间的连接关系。

## 3. 文件结构

```text
FA/
├── cpp_cuda/fa2/
│   ├── binding.cpp
│   ├── fa2_forward.cu
│   └── build_extension.py
├── src/
│   ├── cuda_fa2_forward.py
│   └── cuda_fa2_extension*.so
└── build/
    ├── temp.<platform>/   # .o 中间文件
    └── lib.<platform>/    # 链接前的 .so
```

`build/` 和最终 `.so` 均属于生成产物，已被 Git 忽略。

## 4. 扩展模块工作原理

### 4.1 什么是 CPython Extension

普通 Python 模块通常来自 `.py` 文件。CPython Extension 则是实现了 CPython 模块初始化
协议的动态库，在 Linux 中通常表现为 `.so` 文件：

```text
src/cuda_fa2_extension.cpython-311-x86_64-linux-gnu.so
```

执行以下语句时：

```python
import src.cuda_fa2_extension
```

CPython 会：

1. 在 `src/` 中查找与当前 Python 版本和平台匹配的扩展后缀。
2. 使用操作系统动态链接器加载 `.so`，底层机制类似 `dlopen()`。
3. 在动态库中查找模块初始化函数 `PyInit_cuda_fa2_extension`。
4. 调用初始化函数并获得一个普通 Python Module 对象。
5. 将模块缓存到 `sys.modules["src.cuda_fa2_extension"]`。

因此，加载后的 CUDA Extension 对 Python 而言仍然是普通模块，只是函数实现来自编译后的
C++/CUDA，而不是 `.py` 文件。

### 4.2 两个源码文件如何组成一个 `.so`

构建过程包含三个阶段：

```text
binding.cpp
    └── Host C++ Compiler ──> binding.o

fa2_forward.cu
    └── nvcc ───────────────> fa2_forward.o

binding.o + fa2_forward.o + PyTorch/CUDA Libraries
    └── Host Linker ─────────> cuda_fa2_extension*.so
```

[binding.cpp](../../cpp_cuda/fa2/binding.cpp#L1-L14) 只声明 `fa2_forward_cuda`：

```cpp
torch::Tensor fa2_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal);
```

函数定义位于 [fa2_forward.cu](../../cpp_cuda/fa2/fa2_forward.cu#L115-L143)。编译
`binding.cpp` 时只需要知道函数签名；最终链接阶段再从 `fa2_forward.o` 中解析函数符号。

这里的 `torch::Tensor` 与 PyTorch C++ API 中的 `at::Tensor` 表示同一种引用计数 Tensor
句柄，因此声明使用 `torch::Tensor`、实现使用 `at::Tensor` 仍能链接到同一个函数签名。

当前扩展只有一个绑定文件和一个 CUDA 文件，使用前置声明足够清晰。源码继续拆分后，应将
共享声明放入独立头文件，避免多个文件重复声明。

### 4.3 `torch/extension.h` 提供什么

```cpp
#include <torch/extension.h>
```

该头文件组合了当前绑定需要的主要能力：

- PyTorch C++ Tensor API。
- Pybind11 模块与函数绑定接口。
- Python `torch.Tensor` 与 C++ `torch::Tensor` 之间的类型转换器。
- PyTorch C++ 异常向 Python 异常的转换支持。

有了 Tensor 类型转换器，Python 调用扩展时不会复制 Tensor 数据。Pybind11 只把 Python
Tensor 对象转换成引用同一底层 Storage 的 C++ Tensor 句柄。

### 4.4 `TORCH_EXTENSION_NAME` 从哪里来

绑定代码使用：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    // ...
}
```

构建脚本配置：

```python
CUDAExtension(name="src.cuda_fa2_extension", ...)
```

`BuildExtension` 在编译 `binding.cpp` 时自动添加：

```text
-DTORCH_EXTENSION_NAME=cuda_fa2_extension
```

因此宏展开后的模块名与 `.so` 的 Python 模块名保持一致。Pybind11 进一步生成 CPython
要求的初始化入口，其效果可概念化为：

```cpp
PyObject* PyInit_cuda_fa2_extension();
```

如果编译时模块名与 Python import 名不一致，动态库虽然可能成功链接，但 CPython 会因找不到
正确的 `PyInit_*` 符号而导入失败。

### 4.5 `PYBIND11_MODULE` 做了什么

当前绑定：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &fa2_forward_cuda,
      "FP32 online-softmax attention forward baseline (CUDA)");
}
```

各参数含义：

| 表达式 | 作用 |
|---|---|
| `TORCH_EXTENSION_NAME` | 当前 Extension 模块名 |
| `module` | Pybind11 提供的模块对象 |
| `"forward"` | Python 侧暴露的函数名 |
| `&fa2_forward_cuda` | 绑定的 C++ 函数指针 |
| 最后一个字符串 | `help()` 和 `__doc__` 显示的函数说明 |

模块初始化阶段只注册函数，不会启动 CUDA Kernel。真正的 Kernel Launch 发生在 Python 调用
`extension.forward(...)` 之后。

### 4.6 一次 Forward 的完整调用链

```text
Python
  cuda_fa2_attention_forward(q, k, v)
    │
    ├── import src.cuda_fa2_extension
    │     └── 首次导入时加载 .so 并执行 PYBIND11_MODULE
    │
    └── extension.forward(q, k, v, is_causal)
          │
          ├── Pybind11：Python Tensor -> C++ Tensor 句柄
          ├── fa2_forward_cuda：校验输入并分配输出
          ├── 获取 PyTorch Current CUDA Stream
          ├── fa2_forward_fp32_kernel<<<...>>>()
          └── Pybind11：C++ Tensor 句柄 -> Python Tensor
```

需要区分三类操作：

- **模块导入**：首次执行 `import` 时加载 `.so`；后续由 `sys.modules` 直接复用。
- **Host 调用**：Pybind11 参数转换、C++ 输入校验、输出 Tensor 分配和 Kernel Launch。
- **Device 执行**：CUDA Kernel 在 PyTorch 当前 Stream 上异步运行。

Python Tensor 与 C++ Tensor 的转换不包含 CPU/GPU 数据拷贝。输入 Tensor 的 Device Pointer
由 C++ Wrapper 直接传给 CUDA Kernel。

CUDA Kernel Launch 是异步操作。C++ 函数可以在 Kernel 完成前返回输出 Tensor；后续同一
CUDA Stream 上的算子依靠 Stream 顺序保证数据就绪。Benchmark 中的 CUDA Event 或
`torch.cuda.synchronize()` 才负责等待实际执行完成。

### 4.7 异常与对象生命周期

PyTorch Tensor 采用引用计数管理底层 Storage。Python Tensor 被转换为 C++ Tensor 时，
引用计数会保证输入 Storage 在 C++ 调用期间保持有效；返回的 C++ Tensor 也会转换为受 Python
管理的 Tensor 对象。

C++ 中的 `TORCH_CHECK` 会抛出 PyTorch C++ 异常，Pybind11/PyTorch Binding 将其转换为
Python `RuntimeError`。因此输入 Device、Shape、Dtype 或连续性错误可以直接在 Python 中捕获。

Pybind11 默认在调用绑定函数期间持有 Python GIL。当前 Host Wrapper 只做少量校验、分配和
异步 Kernel Launch，因此暂未主动释放 GIL；如果未来 Host 侧包含长时间同步工作，再评估
`py::call_guard<py::gil_scoped_release>()`。

### 4.8 当前方案与生产级 PyTorch Operator 的区别

当前使用的是直接 Pybind11 绑定：

```text
Python Module -> extension.forward -> C++ Function -> CUDA Kernel
```

该方式适合教学、原型和单一 CUDA 实现，调用链短且容易理解。但它不会自动接入 PyTorch
Dispatcher 的完整能力。

| 能力 | 当前 Pybind11 直连 | 生产级 Custom Operator |
|---|---|---|
| Python 调用 | `extension.forward(...)` | `torch.ops.<namespace>.<op>(...)` |
| Device Dispatch | 手动检查 | Dispatcher 按 Backend 分发 |
| Autograd | 当前未注册 | 可注册 Autograd Kernel |
| Fake/Meta Tensor | 未注册 | 可支持 Shape Inference |
| Autocast | 未注册 | 可注册 Autocast 规则 |
| `torch.compile` | 支持有限 | 可通过 Custom Op 接口集成 |
| 实现复杂度 | 低 | 较高 |

成熟 PyTorch Extension 常使用 `TORCH_LIBRARY`、`TORCH_LIBRARY_IMPL` 或
`torch.library.custom_op` 注册算子，再通过 `torch.ops` 调用。Pybind11 仍常用于原型、
内部工具以及不需要 Dispatcher 集成的薄绑定层。

本项目当前目标是建立可测量的 CUDA Forward 基线，因此保留直接 Pybind11 方案。等到需要
Backward、`torch.compile`、多设备分发或 Autocast 时，再升级为 Dispatcher Custom Operator。

## 5. CUDA Toolkit 探测

对应代码：

```python
def _configure_cuda_home() -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return
    cuda_home = Path(nvcc).resolve().parents[1]
    if (cuda_home / "include" / "cuda_runtime.h").is_file():
        os.environ["CUDA_HOME"] = str(cuda_home)
```

执行过程：

1. `shutil.which("nvcc")` 在当前 `PATH` 中查找 CUDA 编译器。
2. `Path.resolve()` 解析符号链接，避免将 Conda 环境中的 `nvcc` 链接误判为 CUDA Toolkit 根目录。
3. 从 `<cuda-home>/bin/nvcc` 向上两级得到 CUDA 根目录。
4. 检查 `include/cuda_runtime.h`，确认该目录确实包含 CUDA 开发文件。
5. 设置 `CUDA_HOME`，供 PyTorch 的 `CUDAExtension` 查找头文件和运行库。

如果找不到 `nvcc`，函数直接返回，后续 `CUDAExtension` 会给出标准的 CUDA Toolkit 缺失错误。

## 6. 为什么延迟导入 PyTorch 构建模块

对应代码：

```python
def main() -> None:
    _configure_cuda_home()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

`torch.utils.cpp_extension` 会在导入时解析 `CUDA_HOME`。因此必须先执行
`_configure_cuda_home()`，再导入 `BuildExtension` 和 `CUDAExtension`。

如果把这些 import 放在文件顶部，PyTorch 可能在 `CUDA_HOME` 修正之前缓存错误的 Toolkit 路径。

## 7. `setup()` 配置

### 7.1 Distribution 名称

```python
name="cs336-cuda-fa2"
```

这是 Setuptools 使用的 Distribution 名称，主要出现在构建日志和元数据中，不是 Python import 名称。

### 7.2 Python Package

```python
PROJECT_ROOT = SOURCE_DIR.parents[1]

os.chdir(PROJECT_ROOT)

packages=["src"]
package_dir={"src": "src"}
```

这里的 Python Package 指可通过 `import` 访问的目录，不表示需要将项目发布到 PyPI。
由于 `src/` 中存在 `__init__.py`，因此它对应 Python 导入路径中的 `src`。

路径基准按以下步骤确定：

1. `SOURCE_DIR` 是 `build_extension.py` 所在的 `FA/cpp_cuda/fa2/`。
2. `SOURCE_DIR.parents[1]` 得到 `FA/` 项目根目录。
3. `os.chdir(PROJECT_ROOT)` 将构建进程的工作目录固定为 `FA/`。
4. `package_dir={"src": "src"}` 显式声明：Python 包名 `src` 对应当前基准目录下的 `src/`。

因此这里的 `"src"` 最终始终解析为：

```text
<PROJECT_ROOT>/src
= <FA>/src
```

即使从其他目录调用构建脚本，脚本也会先切换到 `FA/`，不会使用调用者原来的工作目录解析
`src`。

扩展完整模块名为：

```text
src.cuda_fa2_extension
```

模块名与文件位置的对应关系为：

```text
src.cuda_fa2_extension
│   └── Python Package: src
└────── Extension 模块: cuda_fa2_extension

FA/src/cuda_fa2_extension.cpython-311-x86_64-linux-gnu.so
```

不使用 `--inplace` 时，链接生成的 `.so` 只保留在 `FA/build/lib.<platform>/src/`
中。使用 `--inplace` 后，Setuptools 根据 `package_dir` 提供的目录映射，将 `.so`
额外复制到 `FA/src/`，因此可以直接执行：

```python
import src.cuda_fa2_extension
```

这不会把 `cpp_cuda/` 变成 Python Package；该目录始终只保存原生源码与构建文档。

### 7.3 Extension 模块名称

```python
name="src.cuda_fa2_extension"
```

该名称同时决定：

- Python import 路径：`src.cuda_fa2_extension`。
- 最终 `.so` 的目标目录：`src/`。
- `PYBIND11_MODULE` 中 `TORCH_EXTENSION_NAME` 展开的模块名：`cuda_fa2_extension`。

Python Wrapper 与这里的名称必须保持一致。

### 7.4 源文件

```python
sources=[
    str(SOURCE_DIR / "binding.cpp"),
    str(SOURCE_DIR / "fa2_forward.cu"),
]
```

PyTorch 根据扩展名选择编译器：

| 文件 | 编译器 | 作用 |
|---|---|---|
| `binding.cpp` | Host C++ Compiler | Pybind11 绑定 |
| `fa2_forward.cu` | `nvcc` | CUDA Kernel 与 Launch Wrapper |

使用基于 `__file__` 的绝对路径后，只要从 `FA` 目录执行，构建过程就不会依赖源码文件的相对查找行为。

## 8. 编译参数

```python
extra_compile_args={
    "cxx": ["-O2"],
    "nvcc": ["-O2", "-lineinfo"],
}
```

### `-O2`

对 Host C++ 和 CUDA 代码启用常规编译器优化。当前项目优化的是 Kernel 算法与并行方式，
没有必要通过 `-O0` 人为放大编译器层面的低效。

### `-lineinfo`

在 CUDA Binary 中保留行号信息，使 Nsight Systems / Nsight Compute 能将 Kernel 指令
关联回 [fa2_forward.cu](../../cpp_cuda/fa2/fa2_forward.cu)。

它不会像 `-G` 那样生成完整 Device Debug 信息，因此对性能影响较小。

## 9. 为什么不使用 Ninja

```python
BuildExtension.with_options(use_ninja=False)
```

`BuildExtension` 默认优先使用 Ninja，以获得并行和增量编译能力。这里显式关闭 Ninja，
改用 Setuptools / Distutils 的普通构建流程。

两种方式生成的 CUDA Kernel 没有算法或运行时差异；区别只存在于编译调度阶段。

当前扩展只有一个 `.cpp` 和一个 `.cu` 文件，普通串行编译足以满足需求。后续源码规模扩大、
编译耗时成为实际问题后，再评估是否引入 Ninja。

## 10. `main` 入口

```python
if __name__ == "__main__":
    main()
```

只有直接执行 `build_extension.py` 时才启动构建。其他 Python 模块即使导入该文件，
也不会意外触发编译。

## 11. 推荐构建命令

在 `FA` 目录执行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=6.1 \
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py build_ext --inplace
```

参数说明：

| 参数 | 含义 |
|---|---|
| `CUDA_HOME` | 指定 CUDA Toolkit 根目录 |
| `TORCH_CUDA_ARCH_LIST=6.1` | 只生成 GTX 1060（SM 6.1）对应的 Cubin |
| `-E` | 忽略 `PYTHON*` 环境变量 |
| `-s` | 禁用用户级 Site Packages |
| `-B` | 不生成 `.pyc` |
| `build_ext` | 执行 C/CUDA Extension 构建 |
| `--inplace` | 将 `.so` 复制到 `src/` |

## 12. 增量构建与清理

Setuptools 根据源码和 Object 文件的时间戳决定是否重新编译。

源码修改后，通常直接重新执行构建命令即可。需要强制重新编译时：

```bash
python cpp_cuda/fa2/build_extension.py build_ext --inplace --force
```

需要完全清理生成产物时：

```bash
rm -rf build/
rm -f src/cuda_fa2_extension*.so
```

清理后必须重新构建，才能运行 `cuda_fa2`。

## 13. ABI 与重建条件

生成的 `.so` 与构建环境绑定。发生以下变化时应重新构建：

- Python 版本变化。
- PyTorch 版本变化。
- PyTorch CUDA Build 版本变化。
- C++ ABI 配置变化。
- CUDA Toolkit 或 Host Compiler 变化。
- `TORCH_CUDA_ARCH_LIST` 变化。
- C++/CUDA 源码或编译参数变化。

`.so` 文件名中的 `cpython-311` 表示它只能由兼容的 CPython 3.11 环境加载。

## 14. 当前环境警告

当前环境为：

```text
PyTorch CUDA Build: 12.4
CUDA Toolkit:       12.8
GPU:                GTX 1060 / SM 6.1
```

构建时会出现 CUDA 12.8 与 PyTorch CUDA 12.4 的次版本不一致警告。PyTorch 将其标记为
通常可兼容的 Minor Version Mismatch，当前扩展也已成功编译并加载。

为了减少环境差异，长期应优先让 PyTorch CUDA Build 与本地 CUDA Toolkit 保持一致。

## 15. 构建复杂度与并行性

当前只有两个 Translation Unit：

- 一个 Host C++ 文件。
- 一个 CUDA 文件。

关闭 Ninja 后，构建基本串行执行。构建时间主要由 `nvcc` 编译 CUDA Translation Unit
决定；内存占用和 CPU 并发较低，适合作为初始开发方案。

构建方式不会改变 Forward 的运行性能。Kernel 性能只受 CUDA 源码、编译参数、输入 Shape
和 GPU 硬件影响。
