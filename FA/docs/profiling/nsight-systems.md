# Attention Forward Profiling：单次采集与 kernel 时间线

## 1. 目标与采集范围

使用 [scripts/profile_forward.py](../../scripts/profile_forward.py) 采集二维 attention 的 GPU kernel
执行顺序、耗时和 launch 间隙。输入 `Q/K/V` 均为 `(S, d)`，固定使用 **FP32**。
通过 `--impl native|efficient` 选择实现，默认 `native`，没有精度切换参数。

默认使用 `S=16384, d=64`，与 benchmark 默认形状一致。FP32 的分数与概率矩阵各占
`16384² × 4 = 1 GiB`，合计 **2 GiB（约 2.15 GB）**。相较 `S=4096`，
两者存储量和主要算术量均增为 16 倍，便于观察后续优化效果。
这是原生路径两张矩阵的理论显存占用。
memory-efficient 后端不完整物化这两张矩阵，不能用该理论值表示其占用。
实际峰值还包含输入、输出、算子临时量等，
不等于 `nvidia-smi` 的进程总显存。实际耗时增长以测量为准。

**正式采集固定为一次 forward，warmup 不进入时间线。** 没有 `--iterations` 参数。
本入口先用 if/elif 选择 callable，预热和采集复用同一实现，通过 NVTX 标注算子。
优化路径使用 PyTorch 内置 SDPA，只启用指定后端，不支持就报错，不自动回退或转换精度。
正式性能比较见 [benchmark 文档](../benchmark/benchmark.md)。

## 2. 环境与脚本参数
本机已有以下工具，无需重复安装：

| 工具 | 版本 / 用途 |
|---|---|
| `nanovllm` conda 环境 | Python 3.11、PyTorch `2.6.0+cu124` |
| `nsys` / `nsys-ui` | Nsight Systems `2024.6.2`，采集与查看时间线 |
| `nvprof` / `nvvp` | CUDA 12.8 附带的旧版采集器与可视化工具 |

优先使用 Nsight Systems。下面直接指定 `nanovllm` 的 Python，避免项目 `.venv`
抢占 PATH。若使用 `conda run -n nanovllm python ...`，应先用 `deactivate`
退出已激活的项目 venv。

脚本参数由 [parse_args()](../../scripts/profile_forward.py#L17-L32)
独立解析和校验：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--impl` | `native` | `native` 原生基线；`efficient` FP32 memory-efficient SDPA |
| `--seq-len` | `16384` | 序列长度 `S` |
| `--head-dim` | `64` | 特征维度 `d` |
| `--causal` | 关闭 | 添加该参数后启用 causal mask |
| `--warmup` | `5` | 录制前的预热次数 |
| `--threads` | `1` | PyTorch CPU 线程数，不是 GPU 线程数 |
| `--seed` | `0` | 随机种子 |

## 3. 用 Nsight Systems 采集

### 3.1 原生 FP32 基线

在已确认 CUDA 可用的用户终端中执行：
采集报告及后续生成的 SQLite 分析文件统一放在 `FA/profiles/` 独立目录中。

文件名前缀使用 `baseline_naive_attention`，明确表示最原始的 PyTorch attention
基线：完整物化分数和概率矩阵，作为后续自写 kernel 与优化版本的对照。
后续优化版本使用各自的实现前缀，保留这份原始基线。

```bash
cd /home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA

mkdir -p ./profiles

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=./profiles/baseline_naive_attention_s16384_d64 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_forward \
  --impl native --seq-len 16384 --head-dim 64 --warmup 5
```

采集结果为 `FA/profiles/baseline_naive_attention_s16384_d64.nsys-rep`。
重复采集时在文件名后追加运行编号，避免覆盖已有结果。
要看 causal，给 Python 脚本追加 `--causal`，并将文件名前缀设为
`baseline_naive_attention_s16384_d64_causal`。
命令中的反斜杠 `\` 表示下一行仍属于同一条命令，需要将整段复制执行。
Python 的 `-E -s -B` 分别忽略 `PYTHON*` 环境变量、禁用用户级 site-packages、
禁止生成字节码缓存。

### 3.2 录制边界

采集流程位于 [main()](../../scripts/profile_forward.py#L35-L81)：

1. 选择实现，创建输入，检查当前硬件、精度和布局是否支持指定后端。
   预检在采集区间外；通过后执行 5 次 warmup，完成相关 CUDA/kernel 初始化。
2. 同步后调用 `cudaProfilerStart`，此时 Nsight 才开始录制。
3. 只录制一次 forward，标注为 `attention_forward/<impl>/float32`，
   内部用 `emit_nvtx` 自动标注 PyTorch 算子。
4. 所有 GPU 工作完成后调用 `cudaProfilerStop`，结束录制。

`--sample=none --cpuctxsw=none` 关闭本次不需要的 CPU 采样和调度跟踪。
本入口只在采集边界同步，不在各 kernel 之间插入同步。
单独运行这个 Python 脚本不会自动生成报告，必须由 `nsys` 或 `nvprof` 包裹启动。

### 3.3 FP32 成熟优化后端

`efficient` 强制使用 `SDPBackend.EFFICIENT_ATTENTION`，即 PyTorch 内置的
memory-efficient SDPA。PyTorch CUDA FlashAttention 不支持 FP32，因此本实验使用
`efficient` 作为成熟优化参照，保持输入、输出与原生基线均为 FP32。

在 GTX 1060 上可尝试此路径；实际是否支持由当前 PyTorch 构建、设备和输入共同决定。
与第 3.1 节报告对比时，保持 `S/d/causal/warmup` 相同，两条入口均关闭 TF32。

以下命令继续在 `FA` 目录执行：

```bash
mkdir -p ./profiles

nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=./profiles/memory_efficient_attention_s16384_d64_fp32 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_forward \
  --impl efficient --seq-len 16384 --head-dim 64 --warmup 5
```

脚本在采集前打印实现、后端、PyTorch/CUDA 版本、GPU、compute capability、shape 和 dtype。
预检或正式执行发现不支持时会报错；不把 math、cuDNN 或其他后端的结果混入所选实现。
维度适配和严格选择见
[src/backends.py](../../src/backends.py#L23-L50)。

## 4. 查看时间线与耗时

### 4.1 在 GUI 中查看

在有图形界面的机器上打开：

```bash
nsys-ui ./profiles/baseline_naive_attention_s16384_d64.nsys-rep
```

后续命令均在第 3.1 节的 `FA` 目录执行，直接使用 `./profiles/` 路径。
如果当前 Linux 是无桌面的远程机器，将 `.nsys-rep` 下载到装有 Nsight Systems GUI
的电脑，通过 **File → Open** 打开；建议 GUI 与采集端使用相同版本。

展开 GPU 下的 CUDA context / streams 时间线。每个 kernel 是一个横条：
横坐标是时间，横条宽度是执行时长；选中横条后可查看 kernel 名称、开始时间、
duration、stream 和 launch 配置。按进程/线程展开 NVTX 行，可看到
单个 `attention_forward/native/float32`（或 `efficient/float32`）区间与内部算子标签。

CPU 的 CUDA API 行表示 launch 等主机调用；GPU stream 行才表示 kernel 的实际执行。
CPU NVTX 区间结束时，异步发射的 GPU 工作可能还没完成，所以不能把 NVTX 区间长度
直接当成 kernel 耗时。可通过 CUDA API 与 GPU kernel 的关联信息定位对应关系。

当前代码使用一个 CUDA stream，同一 stream 内的 kernel 按顺序执行。
多个 stream 的阶梯状并发需要额外设计。

### 4.2 在终端查看汇总与逐次执行记录

```bash
# 按 kernel 名称聚合：总时间、调用次数、平均值等
nsys stats --report cuda_gpu_kern_sum \
  ./profiles/baseline_naive_attention_s16384_d64.nsys-rep

# 按时间列出每次 GPU kernel / memcpy 等活动
nsys stats --report cuda_gpu_trace \
  ./profiles/baseline_naive_attention_s16384_d64.nsys-rep
```

`stats` 首次运行可能生成同目录的 SQLite 分析文件。注意表头时间单位。
`cuda_gpu_kern_sum` 的时间占比以 kernel 执行时间总和为分母，不等于整个 forward 的
wall-clock 占比。看 launch 间隙、CPU 等待或多 stream 重叠，需要结合时间线。

### 4.3 将 kernel 对应到代码

对于 `native` non-causal forward，可沿算子标签寻找以下工作：

| 代码阶段 | 可能看到的算子标签 / GPU 工作 |
|---|---|
| `q @ k.T` | `aten::mm`；cuBLAS GEMM kernel |
| 除以 `sqrt(d)` | `aten::div`；逐元素缩放 kernel |
| `softmax(s)` | `aten::_softmax`；softmax / reduction kernel |
| `p @ v` | `aten::mm`；另一次 GEMM |
| 开启 causal 时 | 额外的索引生成、比较、mask/copy 等工作 |

实际 kernel 名称和数量取决于 PyTorch、cuBLAS、GPU、shape；一个算子不保证只启动
一个 kernel，也有纯视图操作不启动 kernel。

`efficient` 路径可寻找 `aten::_scaled_dot_product_efficient_attention`。
该路径将分块矩阵乘与 softmax 融合，不应期待与 native 一样独立的
`mm → div → softmax → mm` 序列；GPU kernel 的实际名称和数量仍以报告为准。

### 4.4 追踪显存分配变化

Nsight Systems 支持 `--cuda-memory-usage=true`，在时间线上展示
**CUDA GPU Memory Allocation Graph**。它跟踪 CUDA 层的 GPU 内存分配和释放，
不是逐个 PyTorch Tensor 的生命周期，也不是显存读写带宽。

PyTorch 使用缓存分配器：Tensor 释放后，内存通常留在池中供后续复用，不立即交还
CUDA。因此，原来的 5 次 warmup 可能已完成大部分底层显存分配，正式采集时即使
`S/P` 正常创建和释放，Nsight 显存曲线仍可能很平，或没有新的分配事件。

要观察本次大矩阵向 CUDA 申请显存的过程，可在 `FA` 目录单独运行：

```bash
mkdir -p ./profiles

nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=./profiles/baseline_naive_attention_s16384_d64_memory \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_forward \
  --impl native --seq-len 16384 --head-dim 64 --warmup 0

nsys-ui ./profiles/baseline_naive_attention_s16384_d64_memory.nsys-rep
```

仍然只录一次 forward。这里设置 `--warmup 0` 是为了观察首次大矩阵分配，
会包含首次执行的初始化开销；加上显存跟踪本身的开销，这份报告用于看内存变化，
不用于记录正式加速比。输入 `Q/K/V` 在捕获区间之前已创建，不能仅凭区间内的新分配
事件推断整个进程的总显存。
如果专门衡量显存跟踪开销，普通报告与显存报告必须使用相同的 warmup；
优化效果的正式比较保留 `--warmup 5`。

在 GUI 中展开 GPU 的 memory allocation / memory usage 曲线，检查分配和释放事件。
即使无 warmup，`S/P` 生命周期结束也可能不产生 CUDA free，曲线不一定随 Tensor
释放而下降。

| 观察口径 | 对应工具 / 接口 |
|---|---|
| CUDA 层分配与释放随时间变化 | Nsight 的 GPU Memory Allocation Graph |
| PyTorch 当前活跃分配 | `torch.cuda.memory_allocated()` |
| PyTorch 内存池总占用，含空闲缓存 | `torch.cuda.memory_reserved()` |
| 一次 forward 的 PyTorch 额外峰值 | `scripts.benchmark` 的 `peak_additional_cuda_allocated_bytes` |

若要定位具体 Tensor 的分配和释放，可进一步使用 PyTorch memory profiler；
Nsight 的曲线不能直接解释为 `S/P` 两张 Tensor 的占用曲线。
说明已核对本机 2024.6.2 帮助与
[NVIDIA 官方文档](https://docs.nvidia.com/nsight-systems/2024.6/UserGuide/index.html#cuda-gpu-memory-allocation-graph)。

## 5. 使用旧版 nvprof / nvvp

本机仍提供旧工具，可以尝试相同的限定采集入口。在第 3.1 节的 `FA` 目录执行：

```bash
mkdir -p ./profiles

nvprof --profile-from-start off \
  --export-profile ./profiles/baseline_naive_attention_s16384_d64.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_forward \
  --impl native --seq-len 16384 --head-dim 64 --warmup 5

nvvp ./profiles/baseline_naive_attention_s16384_d64.nvvp
```

`nvvp` 是旧版 NVIDIA Visual Profiler；`.nvvp` 与 `.nsys-rep` 分别使用对应 GUI。
后续迁移到更新 GPU 时，优先沿用 Nsight Systems 流程。

## 6. 结果用途与验证状态

用时间线回答“启动了哪些 kernel、顺序如何、哪里存在空隙、哪个 kernel 累计耗时高”。
记录正式加速比时，仍使用不带 profiler 的 `scripts.benchmark`；NVTX 标注和 profiler
采集都会引入额外开销，profile 内的耗时不应直接替代基准结果。

已核对本机工具版本、采集参数和报告名，脚本语法、帮助及参数校验已通过。
原生 CPU 数值测试和后端调度契约测试通过；调度测试不等于真实 GPU kernel 数值验证。
可先执行 `python -m tests.test_backends --device cuda`，检查本机 FP32 efficient 的数值；
不支持时明确跳过。
代理侧 GPU 设备访问仍受沙箱限制，尚未在代理侧实际生成 `.nsys-rep` 或 `.nvvp`；
请在用户已确认 CUDA 可用的终端执行上述采集命令。
