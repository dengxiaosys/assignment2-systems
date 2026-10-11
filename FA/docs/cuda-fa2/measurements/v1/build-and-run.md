# 1. V1 编译、正确性验证与性能采集

V1 的开发源码位于 [cpp_cuda/fa2/v1/fa2_forward.cu](../../../../cpp_cuda/fa2/v1/fa2_forward.cu)，使用独立模块 `src.cuda_fa2_v1_extension`。既有 `cuda_fa2` 继续加载 V0 扩展，新增 `cuda_fa2_v1` 用于 V1；两版可在同一 Python 环境中运行。设备、launch 和 shared memory 背景见 [初学者 FAQ](faq.md)，算法原理与优化前预期见 [优化设计](optimization-design.md)，本次测量结果见 [V1 性能与 profile 分析](perf_v1.md)。

## 1.1 源码与产物位置

```text
FA/
├── cpp_cuda/fa2/
│   ├── build_extension.py          统一编译入口：--version v0 / v1
│   ├── common/
│   │   └── binding.cpp             共用 pybind11 forward 绑定
│   ├── v0/
│   │   └── fa2_forward.cu          V0 核心实现
│   └── v1/
│       └── fa2_forward.cu          V1：一 warp 一行，四行 / CTA
├── src/
│   ├── cuda_fa2_extension*.so       原 V0 实际扩展
│   ├── cuda_fa2_v1_extension*.so    V1 开发运行扩展
│   └── cuda_fa2_forward.py          两版加载、调用及路径 / SHA-256 记录
├── scripts/
│   ├── benchmark_attention.py      --impl cuda_fa2_v1
│   └── profile_attention.py        --impl cuda_fa2_v1
├── tests/
│   └── test_cuda_fa2_forward.py    常规验证及可选长序列抽样检查
└── docs/cuda-fa2/measurements/v1/
    ├── faq.md                      设备、launch 与 shared memory 问答
    ├── build-and-run.md             本文：复现操作
    ├── optimization-design.md       优化前模型与假设
    ├── perf_v1.md                   V1 实测与解释
    ├── manifest.json               源码、二进制、环境、样本及数据身份
    ├── snapshot/fa2_forward.cu      本轮核心 CUDA 实现快照
    ├── binaries/                   实际被测扩展
    ├── data/                       编译日志、验证、benchmark、CSV、SASS
    ├── profiles/                   profiler 原始报告
    └── assets/                     优化设计 SVG
```

开发时修改 `cpp_cuda/fa2/v1/`，然后通过统一入口选择 V1 构建。`snapshot/` 只固定核心实现，公共绑定、构建脚本、Python 运行入口和测试直接引用工作区；实际被测扩展单独放在 `binaries/`，构建参数、日志与哈希由 manifest 保存。`.cu` 不会在导入时自动编译，构建后需使用新 Python 进程。版本选择由 [forward registry](../../../../src/forward_registry.py) 完成；[扩展身份记录函数](../../../../src/cuda_fa2_forward.py#L88-L100) 提供 `extension_module/extension_path/extension_sha256`，并在计时和 capture 范围之外计算哈希。

## 1.2 编译环境与资源控制

本次环境为 GTX 1060、CC 6.1，Python 环境 `nanovllm`，PyTorch `2.6.0+cu124`，CUDA 编译器 `12.8.93`，驱动 `570.211.01`，g++ `13.3.0`。CUDA 12.8 与 PyTorch 编译版本 12.4 存在 minor mismatch，构建日志保留该提示；实际扩展编译、加载与 CUDA 验证均成功。这里固定 `TORCH_CUDA_ARCH_LIST=6.1`，使用 FP32 SIMT，不启用 `--use_fast_math`。

以下命令在 `FA` 目录执行。环境变量中 CPU 并行度为 1；编译中间文件放入 `/tmp/fa2-v1-build/`，最终扩展按 `--inplace` 放入 `src/`。`env -u` 清除当前 IDE 注入的动态库路径，避免系统 CUDA 工具误加载不匹配的 `libstdc++`。

```bash
cd /home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py --version v1 build_ext --inplace \
  --build-temp /tmp/fa2-v1-build/temp \
  --build-lib /tmp/fa2-v1-build/lib \
  > /tmp/fa2-v1-build.log 2>&1
```

统一 [编译脚本](../../../../cpp_cuda/fa2/build_extension.py) 根据 `--version` 选择源码和独立模块名，默认 V1；两个版本共用 `common/binding.cpp`。编译参数为 `-O2`、`-lineinfo` 和 `--ptxas-options=-v`，最后一项输出寄存器、barrier 与 spill 信息。若强制重新编译相同路径，可给 `build_ext` 增加 `--force`。本轮原始构建日志中的旧路径属于整理前命令，保留其原文；上述命令使用当前统一布局。

长时间 benchmark 或 replay 可在 `tmux` 中执行：

```bash
tmux new -s fa2-v1
```

在会话中按下面步骤依次运行；CPU 线程保持为 1，GPU 的 benchmark 与 profiler 不并发执行。断开后用 `tmux attach -t fa2-v1` 恢复。

## 1.3 正确性验证

### 1.3.1 小形状、全维度与因果语义

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v1
```

[验证入口](../../../../tests/test_cuda_fa2_forward.py) 支持选择 V0/V1，覆盖 \(d=1,\ldots,128\)、条带边界、不足四行的 CTA、causal/non-causal、uniform attention、未来 K/V 不影响 causal 前缀、max 多次改变、输入只读与输出属性。CPU FP64 参考的 `rtol=atol=2e-5` 保持不变。本次输出见 [CUDA 验证日志](data/cuda-validation.log)。

### 1.3.2 长序列抽样行

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward \
  --device cuda --impl cuda_fa2_v1 --long-sequence \
  --output-json /tmp/fa2-v1-row-validation.json
```

[逐行参考函数与长序列测试](../../../../tests/test_cuda_fa2_forward.py#L145-L189) 已合并到现有测试文件。`--long-sequence` 在常规九项测试之外增加一项长序列测试，固定 `S=16384,d=64`、seed 0、一个 CPU 线程，验证行 `0,1,3,4,8191,16383` 的 causal/non-causal 结果；默认不运行长序列检查。`--output-json` 可保存该项结果，只在全部测试成功后写入。

逐行 FP64 参考不物化完整 \(S^2\) 矩阵：输入副本为 \(O(Sd)\)，单行工作区为 \(O(S+d)\)，六行计算工作为 \(O(6Sd)\)；各参考行可独立执行，本次显式使用一个 CPU 线程。本次原始数据见 [长序列验证 JSON](data/row-validation.json)，抽样通过不等于全部输出行都已与 FP64 比较。

### 1.3.3 内存与同步检查

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v1

env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool synccheck --error-exitcode 99 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v1
```

检查工具是否支持当前 GPU 以实际日志为准。程序退出、工具报错或不支持设备均不能解释为“检查通过”；本轮状态由性能文档与原始日志记录。

## 1.4 无 profiler 的正式 benchmark

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark_attention \
  --device cuda --impl cuda_fa2_v1 --seq-len 16384 --head-dim 64 \
  --warmup 1 --iterations 4 --repeats 3 --threads 1 --seed 0 \
  --no-causal --no-verify
```

`--no-causal` 与 V0 计数器基线保持相同工作量；脚本默认 causal，省略该参数会改变比较口径。`--no-verify` 仅跳过计时入口的全矩阵 FP64 验证，独立验证按 §1.3 执行。正式结果为每组 20 次 forward 的 Event/wall 时间除以 20，再对 5 组计算均值与中位数，所有组样本均保留。

**常规优化直接引用 V0 归档基线，无需重新运行 V0。** 主形状 non-causal 的参考均值为 2922.2971875 ms，来源为 [V0 版本清单](../v0/manifest.json) 的 `unprofiled_benchmark.formal` 和 [原始记录](../v0/data/evidence.json) 的 `benchmark_records`；V1 速度比按该均值除以新版本均值计算。硬件计数器同样引用 [V0 已归档 counters](../v0/data/counters.json)。

需要运行 efficient 对照时将 `--impl` 改为 `efficient` 并单独保存；V1 causal 测量将 `--no-causal` 换成 `--causal`，不能与 non-causal 混算。仅在明确需要校准硬件、工具链或运行条件时另开独立校准实验，V0 复跑不属于每轮优化的固定步骤。

## 1.5 nvprof 计数器与时间线

### 1.5.1 硬件计数器采集

本设备计数器权限为 `RmProfilingAdminOnly: 1`，计数器命令使用 sudo；由终端正常完成认证，不将密码写入脚本或归档。`--kernels` 和 `--aggregate-mode` 位于 `--metrics` 之前，匹配四个模板实例中的实际 capture kernel。下面命令只捕获 warmup 后一次 forward，多指标会触发 replay。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --kernels '.*fa2_forward_fp32_kernel.*' --aggregate-mode on \
  --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,eligible_warps_per_cycle,issue_slot_utilization,stall_sync,stall_memory_dependency,stall_exec_dependency,stall_other,stall_not_selected,stall_inst_fetch,stall_pipe_busy,stall_texture,stall_memory_throttle,stall_constant_memory_dependency,dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_tex_read_hit_rate,dram_read_transactions,l2_read_transactions,gld_efficiency \
  --print-gpu-trace --csv --log-file /tmp/fa2-v1-counters.csv \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention \
  --impl cuda_fa2_v1 --seq-len 16384 --head-dim 64 \
  --warmup 5 --threads 1 --seed 0 --no-causal
```

所有吞吐与 stall 数值来自相应采集 pass，不能用 replay 时长代替 benchmark，也不能把 stall 百分比当成墙钟时间占比。本次 CSV 的 `<OVERFLOW>` 必须保留并从定量解释中排除。

### 1.5.2 不带 counters 的单 kernel 时间线

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --print-gpu-trace --csv --log-file /tmp/fa2-v1-timeline.csv \
  --export-profile /tmp/fa2-v1-forward.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention \
  --impl cuda_fa2_v1 --seq-len 16384 --head-dim 64 \
  --warmup 5 --threads 1 --seed 0 --no-causal
```

时间线核对 grid、block、kernel 数量与时长；独立 benchmark 仍作为速度比依据。GTX 1060 使用这里的 nvprof 路线，通用工具选择见 [profiling 工具说明](../../profiling-tools.md)。

## 1.6 SASS、快照与复测记录

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  src/cuda_fa2_v1_extension.cpython-311-x86_64-linux-gnu.so \
  > /tmp/fa2-v1-resources.txt

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-sass \
  src/cuda_fa2_v1_extension.cpython-311-x86_64-linux-gnu.so \
  > /tmp/fa2-v1-sass.txt
```

按模板符号分别检查，`ILi2E` 对应主配置每 lane 两个维度：无 `BAR.SYNC` 与 `MEMBAR.CTA`，有五级归约、两次权重广播和最终分母广播的 `SHFL`；同一源码原语可能因编译器展开产生多个静态位置，因此其他模板的静态 `SHFL` 数量不应直接当成每 key 动态次数。

每版 `snapshot/` 仅保留实际测量的核心 `.cu`；实际 `.so` 放入 `binaries/`。公共绑定、构建脚本、Python 入口和测试只维护一份，构建日志、flags、环境与源码/二进制 SHA-256 保存在 manifest 和 data 中。V0 的已有源码、计时和 counters 通过相对链接复用。新版本多轮采集可用独立 `run1/`、`run2/`，文件名不带日期；结果解释对照 [优化前 H1–H8](optimization-design.md#performance-hypotheses)，分别判断结构、指标、性能和数值条件。
