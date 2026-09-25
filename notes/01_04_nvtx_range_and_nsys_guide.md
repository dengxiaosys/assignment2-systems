# NVTX Range 与 Nsight Systems 入门

## 0. 本文目标

本文解释下面这行代码的作用及其背后的 profiling 知识：

```python
measurement_range = torch.cuda.nvtx.range("benchmark_measurement") if device.type == "cuda" else nullcontext()
```

代码位置：[benchmark.py:L254](../cs336_systems/benchmark.py#L254)。

读完后应能回答：

1. NVTX 和 Nsight Systems 分别是什么；
2. 为什么 GPU 计时必须考虑异步执行；
3. `torch.cuda.nvtx.range()` 本身做了什么、没有做什么；
4. 为什么 CPU 分支使用 `nullcontext()`；
5. 为什么 warmup 不在 `benchmark_measurement` 内；
6. 如何采集、过滤和分析 `.nsys-rep`；
7. 当前实现还缺少哪些细粒度标记。

---

## 1. 先建立整体认识

### 1.1 Nsight Systems 是什么

NVIDIA Nsight Systems，命令行工具名为 `nsys`，是系统级 timeline profiler。它主要回答：

- CPU 线程在什么时候运行；
- Python/PyTorch 什么时候调用 CUDA Runtime API；
- CUDA kernel 什么时候提交和执行；
- CPU 与 GPU 是否存在空闲、等待或重叠；
- kernel、显存复制和同步操作分别占用多少时间。

它适合看“整段程序如何运行”。如果需要分析单个 kernel 的指令、吞吐率、occupancy 和显存访问效率，通常改用 Nsight Compute，即 `ncu`。

### 1.2 NVTX 是什么

NVTX 全称为 NVIDIA Tools Extension。应用程序可以向 profiling timeline 写入带名称的标记和区间，例如：

```text
benchmark_measurement
├── forward
├── loss
├── backward
└── optimizer
```

NVTX 不负责测量，也不负责优化。它只是给 timeline 加标签，使 profiler 知道某段程序在业务语义上代表什么。

可以把两者理解为：

```text
NVTX：代码主动写入章节标题
nsys：记录完整时间线，并显示这些章节标题
```

### 1.3 CUDA 默认异步执行

CPU 调用 CUDA kernel 时，通常只是把工作提交到 CUDA stream，然后继续执行后面的 Python 代码：

```text
CPU: launch kernel -> immediately continue
GPU:                 -> execute kernel later
```

因此下面的朴素计时通常不准确：

```python
start = time.perf_counter()
output = model(inputs)
elapsed = time.perf_counter() - start
```

`elapsed` 主要包含 CPU 提交 kernel 的时间，不一定包含 GPU 完成计算的时间。

本项目在每个计时阶段末尾调用 [`torch.cuda.synchronize()`](../cs336_systems/benchmark.py#L109-L120)，强制 CPU 等待 GPU 完成已提交工作，从而得到端到端阶段时间。

---

## 2. 逐段解释目标代码

### 2.1 条件表达式

原代码可以展开为：

```python
if device.type == "cuda":
    measurement_range = torch.cuda.nvtx.range("benchmark_measurement")
else:
    measurement_range = nullcontext()
```

它根据运行设备创建一个上下文管理器：

- CUDA：创建名为 `benchmark_measurement` 的 NVTX range；
- CPU：创建什么都不做的 `nullcontext()`。

### 2.2 `torch.cuda.nvtx.range(...)`

```python
torch.cuda.nvtx.range("benchmark_measurement")
```

返回一个 Python 上下文管理器。配合 `with` 使用时，逻辑上等价于：

```python
torch.cuda.nvtx.range_push("benchmark_measurement")
try:
    ...
finally:
    torch.cuda.nvtx.range_pop()
```

进入 `with` 时向当前 CPU 线程压入一个 NVTX range，离开时弹出。即使代码抛出异常，context manager 也会执行退出逻辑，避免 timeline 中留下未闭合区间。

### 2.3 `nullcontext()`

`nullcontext()` 来自 Python 标准库 `contextlib`：

```python
from contextlib import nullcontext
```

它支持 `with` 协议，但进入和退出时都不做任何事：

```python
with nullcontext():
    operation()
```

等价于：

```python
operation()
```

这样 CPU 和 CUDA 可以共享同一套控制流，不需要复制 measurement 循环。

### 2.4 `with measurement_range`

创建 context manager 后，代码在 [benchmark.py:L255-L267](../cs336_systems/benchmark.py#L255-L267) 中使用：

```python
with measurement_range:
    for _ in range(measurement_steps):
        step_timings = _run_step(..., measure=True)
```

当设备为 CUDA 时，Nsight Systems timeline 会显示：

```text
benchmark_measurement
└── measurement step 1
    ├── forward
    ├── loss
    ├── backward
    └── optimizer
```

当前只有最外层 `benchmark_measurement` 标签，内部阶段是根据同步边界离线推断的，并没有直接显示为四个 NVTX 子区间。

---

## 3. Warmup 为什么不在这个 Range 中

代码顺序是：

```python
for _ in range(warmup_steps):
    _run_step(..., measure=False)

with measurement_range:
    for _ in range(measurement_steps):
        _run_step(..., measure=True)
```

对应位置：

- warmup：[benchmark.py:L242-L251](../cs336_systems/benchmark.py#L242-L251)
- measurement：[benchmark.py:L253-L267](../cs336_systems/benchmark.py#L253-L267)

所以 timeline 的逻辑布局为：

```text
process
├── model initialization
├── warmup step 1
├── ...
├── warmup step 5
└── NVTX: benchmark_measurement
    └── measurement step
```

Warmup 仍然会执行 CUDA kernel 和同步，只是不属于 `benchmark_measurement` range。

另一个区别是：

- `measure=False`：整个 warmup step 结束时只同步一次；
- `measure=True`：step 开始、forward、loss、backward、optimizer 和 total 结束处同步。

因此当前 full measurement step 中观察到 6 次 `cudaDeviceSynchronize()`，而每个 full warmup step 只有 1 次。

---

## 4. NVTX Range 不会自动做什么

### 4.1 不会自动同步 GPU

下面的代码不会等待 GPU：

```python
with torch.cuda.nvtx.range("forward"):
    output = model(inputs)
```

NVTX 只标记 CPU 线程上的代码区间。真正保证计时边界正确的是本项目的 `torch.cuda.synchronize()`。

### 4.2 不会自动缩短采集范围

仅在代码中添加 NVTX range：

```python
with torch.cuda.nvtx.range("benchmark_measurement"):
    ...
```

并不意味着 `nsys` 只记录该区间。普通命令：

```bash
nsys profile -- python benchmark.py
```

通常仍会采集进程初始化、warmup、measurement 和退出过程。NVTX range 只是让后续 GUI 或 SQLite 查询可以筛选 `benchmark_measurement`。

### 4.3 不会自动生成阶段统计

当前 range 包围整个 measurement 循环，因此不能直接在 GUI 中点击 `forward` 或 `optimizer`。本项目的单 profile 报告利用连续同步事件划分阶段，详见 [Nsight Systems 单 Profile 实验报告](./01_03_nsys_profile_analysis.md)。

---

## 5. 当前 Benchmark 的时间线

当前使用：

```text
mode=full
warmup_steps=5
measurement_steps=1
```

measurement 内部执行顺序为：

```text
NVTX benchmark_measurement begins
│
├── synchronize before timing
├── forward
├── synchronize
├── loss
├── synchronize
├── backward
├── synchronize
├── optimizer.step
├── synchronize
├── final synchronize
│
NVTX benchmark_measurement ends
```

在现有 GTX 1060 profile 中：

| 范围 | 时间 |
|---|---:|
| `benchmark_measurement` NVTX range | 496.960 ms |
| Python `total` | 495.851 ms |
| CUDA kernel 时长之和 | 482.754 ms |

三者含义不同：

- NVTX range：进入和退出 context manager 之间的主机墙钟区间；
- Python `total`：`_run_step()` 内部显式计时区间；
- kernel 时长之和：GPU 实际执行 kernel 的累计时间。

NVTX range 略长，因为还包含调用 `_run_step()`、收集结果和循环控制等主机端开销。

---

## 6. 如何采集 Profile

### 6.1 当前固定脚本

项目提供：

[run_small_profile.sh](../remote_profile/run_small_profile.sh)

在 `cuda-via-a` 上运行：

```bash
cd ~/work/cs336-profile/assignments/assignment2-systems
./remote_profile/run_small_profile.sh
```

其核心形式是：

```bash
nsys profile \
  --output="$REPORT_BASE" \
  -- python benchmark.py ...
```

`--` 用来分隔：

- 左侧：`nsys profile` 自身的参数；
- 右侧：要被 profiler 启动和跟踪的程序。

### 6.2 为什么没有使用 `uv run nsys`

远端 profile-only 环境没有按 Assignment 2 的 `uv.lock` 创建，因为锁文件指向 GTX 1060 不支持的 CUDA 13.0 PyTorch。本实验直接指定兼容环境的 Python：

```bash
"$HOME/.venvs/cs336-profile-cu126/bin/python" benchmark.py
```

`nsys` 是系统级 NVIDIA 工具，不需要安装进 Python 虚拟环境。

### 6.3 输出文件

当前报告位于 C：

```text
~/var/cs336-profile/profiles/small_b2_s512/profile.nsys-rep
```

复制到开发机后的文件：

```text
assignment2-systems/profile_artifacts/small_b2_s512.nsys-rep
```

`profile_artifacts/` 已加入 `.gitignore`，不会把二进制 profile 提交到 Git。

---

## 7. 如何查看和分析

### 7.1 使用图形界面

在 Nsight Systems GUI 中打开 `.nsys-rep` 后，重点查看：

1. NVTX row：定位 `benchmark_measurement`；
2. CUDA API row：观察 kernel launch 和同步；
3. CUDA GPU row：观察 kernel 的实际执行；
4. CUDA GPU Kernel Summary：按累计 GPU 时间排序；
5. 搜索 `cudaDeviceSynchronize`：定位阶段边界。

### 7.2 使用 `nsys stats`

在 C 上执行：

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  ~/var/cs336-profile/profiles/small_b2_s512/profile.nsys-rep
```

首次运行时，`nsys stats` 会从 `.nsys-rep` 导出 SQLite，然后生成 kernel 汇总。

常见报告包括：

```bash
nsys stats --report cuda_api_sum profile.nsys-rep
nsys stats --report cuda_gpu_kern_sum profile.nsys-rep
nsys stats --report nvtx_sum profile.nsys-rep
```

### 7.3 直接查询 SQLite

导出的 SQLite 中，本实验主要使用：

| 表 | 内容 |
|---|---|
| `NVTX_EVENTS` | NVTX range 的开始、结束和名称 |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA Runtime API 调用 |
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernel 时间、名称和 launch shape |
| `StringIds` | kernel/API 名称字符串 |

分析流程是：

```text
找到 benchmark_measurement 的 start/end
    -> 筛选 range 内的 cudaDeviceSynchronize
    -> 用同步结束时间划分阶段
    -> 分阶段聚合 kernel 时间和次数
```

---

## 8. `--capture-range=nvtx` 与普通采集

Nsight Systems 支持根据 NVTX 控制实际采集：

```bash
nsys profile \
  --capture-range=nvtx \
  --nvtx-capture=benchmark_measurement \
  -- python benchmark.py
```

理论上，这会等到指定 range 开始时才启动采集，从而减少 warmup 和初始化产生的报告体积。

但是要区分：

| 用法 | 行为 |
|---|---|
| 代码中只添加 NVTX range | 完整采集，分析时按 range 过滤 |
| 再指定 `--capture-range=nvtx` | 尝试只在该 range 内开启采集 |

在本机 Nsight Systems 2024.6.2 实验中，NVTX capture-range 命令返回成功，但没有导出 `.nsys-rep`。因此当前固定脚本选择更稳定的完整采集，再通过 `benchmark_measurement` 离线过滤。

---

## 9. 如何增加阶段级 NVTX

当前只有一个外层 range。若希望 GUI 直接显示各阶段，可在 `_run_step()` 中增加嵌套 range：

```python
with torch.cuda.nvtx.range("forward"):
    logits = model(input_ids)

with torch.cuda.nvtx.range("loss"):
    loss = cross_entropy(...)

with torch.cuda.nvtx.range("backward"):
    loss.backward()

with torch.cuda.nvtx.range("optimizer"):
    optimizer.step()
```

期望 timeline：

```text
benchmark_measurement
├── forward
├── loss
├── backward
└── optimizer
```

还可以继续在 attention 中标记：

```text
self_attention
├── qk_matmul
├── mask
├── softmax
└── av_matmul
```

这样可以直接回答“softmax 与 attention matrix multiplication 分别耗时多少”，不再依赖 kernel 名称、grid shape 和调用次数进行离线推断。

实现时仍需保留必要同步，或者明确区分：

- NVTX CPU 区间；
- kernel launch 所属区间；
- GPU kernel 的实际执行区间。

---

## 10. 常见误区

### 10.1 “NVTX range 就是计时器”

不是。NVTX 是标签；计时来自 profiler 事件时间戳或显式 timer。

### 10.2 “有 NVTX range 就不需要同步”

不是。CUDA 异步执行问题仍然存在。端到端 Python 计时必须在边界同步。

### 10.3 “`nsys profile` 只记录 GPU”

不是。Nsight Systems 是系统级 profiler，也可以记录 CPU 线程、OS runtime、CUDA API、NVTX、显存复制和其它库事件。

### 10.4 “所有耗时相加一定等于墙钟时间”

不一定。不同 stream、CPU 线程或设备上的事件可能并行重叠。本实验只有一个 CUDA stream，因此 GPU kernel 时长可以直接相加，但一般不能默认这样做。

### 10.5 “Profiler 中最慢的 kernel 就是最值得优化的代码”

不一定。还要结合：

- 调用次数；
- 单次耗时；
- kernel 是否来自关键路径；
- CPU/GPU 是否并行；
- 优化是否会引入额外同步或显存开销。

---

## 11. 本行代码的最终总结

```python
measurement_range = torch.cuda.nvtx.range("benchmark_measurement") if device.type == "cuda" else nullcontext()
```

它创建一个跨 CPU/CUDA 通用的上下文管理器：

1. CUDA 上向 Nsight timeline 写入名为 `benchmark_measurement` 的 NVTX 区间；
2. CPU 上使用空上下文，不改变程序行为；
3. 外层 `with` 只包围正式 measurement，不包围 warmup；
4. 它本身不计时、不同步，也不会自动限制 `nsys` 的采集范围；
5. 它的主要价值是把正式测量区间从初始化和 warmup 中明确标记出来，便于 GUI 筛选或 SQLite 查询。
