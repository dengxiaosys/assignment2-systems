# GTX 1060 FlashAttention 性能验证与优化报告

## 0. 阅读指南

这份报告回答四个问题：

1. 本项目的 naive attention 和 tiled FlashAttention 分别做了什么？
2. tiled 实现是否真的减少了显存占用，并且比 naive 实现更快？
3. 它与 PyTorch 原生 `scaled_dot_product_attention` 相比处于什么位置？
4. 实测暴露了哪些瓶颈，下一步应该优先优化哪里？

如果只关心结论，可以先看下面这张表。

| 问题 | 结论 |
|---|---|
| Tiled 是否正确？ | 正确。CPU 有限差分、CUDA reference 对比和 Compute Sanitizer 全部通过。 |
| Tiled 是否比 naive 快？ | 是。长序列 forward 快约 `3.4x-3.8x`，backward 快约 `2.5x-2.8x`。 |
| Tiled 是否更省 workspace？ | 是。`N=8192` 时 backward workspace 从约 `1 GiB` 降到 `32 KiB`。 |
| Tiled 是否全面快于 PyTorch？ | 否。它的 backward 快于本机默认 memory-efficient SDPA，但明显慢于使用 cuBLAS 的 PyTorch math backend。 |
| 最大优化收益来自哪里？ | 消除 shared-memory bank conflict，而不是 fast math。 |
| 当前最值得继续优化哪里？ | `dK/dV` 和 `dQ` 两个主 kernel 的点积 microkernel、向量化和 tile shape。 |

这里的“FlashAttention”指算法性质，即按 tile 重建分数和概率、不在显存中
物化完整的二次规模中间矩阵。它不表示本实现达到了 Dao-AILab 官方
FlashAttention CUDA kernel 的工程成熟度。

---

## 1. 实验目标与范围

### 1.1 实验目标

本实验不是只验证一个 CUDA kernel 能否运行，而是建立一条完整的验证链：

```text
CPU reference
    -> naive CUDA
    -> tiled CUDA
    -> PyTorch 原生 SDPA
    -> correctness / memory / latency / profiler 对比
```

具体目标如下：

1. 使用可信的 CPU reference 验证 forward 和 backward 数学公式；
2. 使用容易阅读的 naive CUDA 作为算法基线；
3. 实现不物化完整 `S/P/dP/dS` 的 tiled CUDA；
4. 在 GTX 1060 上验证 Pascal 架构兼容性；
5. 与 PyTorch default、memory-efficient、math 和显式 eager attention 对比；
6. 根据真实 benchmark 和 profiler 结果改进 kernel，而不是只做静态推测。

### 1.2 当前实现范围

| 能力 | 当前状态 |
|---|---|
| 数据类型 | FP32 |
| Attention head | 单 head |
| Tensor 布局 | 连续 row-major，shape 为 `(Nq, d)` 或 `(Nk, d)` |
| Causal mask | 支持，规则为 `key_index <= query_index` |
| Rectangular attention | 支持 `Nq != Nk` |
| Tiled head dimension | `head_dim <= 128` |
| Dropout | 不支持 |
| Batch、多 head、MQA/GQA | 尚未封装 |
| Tensor Core | GTX 1060 不支持，因此未使用 |
| `cp.async` | Pascal 不支持，因此未使用 |
| Triton | 未使用 |

公共接口和约束定义在
[`include/fa/attention.h`](./include/fa/attention.h) 与
[`include/fa/shape.h`](./include/fa/shape.h)。

### 1.3 后文会使用的 GPU 术语

| 术语 | 本报告中的含义 |
|---|---|
| Thread | 一个 CUDA 执行线程。每个 thread 只持有少量局部状态。 |
| Warp | NVIDIA GPU 的基本调度组，由 32 个 thread 组成；warp 内线程通常执行同一条指令。 |
| Block | 一组能共享 shared memory、能使用 `__syncthreads()` 同步的 thread。 |
| Tile | 从大矩阵切出的一小块数据。本实现一次流式处理 32 个 key 或 query。 |
| Register | 每个 thread 私有、速度最快但容量有限的片上存储。 |
| Shared memory | 一个 block 内所有 thread 可访问的片上存储，远快于 global memory，但有 bank conflict 和容量限制。 |
| Global memory | CUDA device memory。GTX 1060 上物理介质是 GDDR5，容量大但访问代价高。 |
| HBM | FlashAttention 论文常用的外部显存统称；GTX 1060 实际不是 HBM，本报告讨论算法时沿用其“片外显存”含义。 |
| Workspace | 除输入、输出和最终梯度外，算法额外申请的临时 device memory。 |
| Materialize | 把逻辑上的中间量完整写成一个真实张量，而不是计算后立即消费。 |
| Owner | 唯一负责累计并写回某一输出行的 warp/block。明确 owner 可避免多个 block 同时写同一位置。 |
| Backend | PyTorch SDPA 在运行时选择的具体实现，如 Flash、memory-efficient、cuDNN 或 math。 |
| Occupancy | 一个 SM 同时驻留的活跃 warp 比例。寄存器和 shared memory 用量都会限制它。 |
| Bank conflict | 一个 warp 的多个 lane 同时访问 shared memory 同一 bank 的不同地址，导致访问被拆分执行。 |

可以把本实现的数据移动层次简化成：

```text
Global memory
    -> 当前 Q/K/V tile
Shared memory
    -> 当前线程需要的标量和 accumulator
Registers
    -> 计算完成后写回 O、dQ、dK、dV
Global memory
```

FlashAttention 的关键不是“完全不访问显存”，而是让 `S/P/dP/dS` 只在 tile
内部短暂存在，避免把完整的 `Nq x Nk` 矩阵反复写入和读出 global memory。

---

## 2. 先理解 Attention 在计算什么

### 2.1 数学记号与实际张量布局

数学上，query 向量 $q_i$、key 向量 $k_j$ 和 value 向量 $v_j$ 都按列向量
理解。单个 attention score 为：

$$S_{ij}=\frac{q_i^\top k_j}{\sqrt d}$$

对第 $i$ 个 query，softmax 概率和输出为：

$$P_{ij}=\frac{\exp(S_{ij})}{\sum_t \exp(S_{it})},\qquad o_i=\sum_j P_{ij}v_j$$

在数学矩阵形式中通常写作：

$$S=\frac{QK^\top}{\sqrt d},\qquad P=\operatorname{softmax}(S),\qquad O=PV$$

数学上的单个向量仍是列向量；代码为了符合 C++/PyTorch 的连续张量布局，
把多个向量按行存为 `(N, d)`，因此内存中的第 `i` 行对应数学向量
$q_i^\top$。这只是存储布局与数学记号的区别，不改变公式含义。

### 2.2 为什么普通 attention 需要二次规模中间量

若 `Nq = Nk = N`，完整 score 矩阵 `S` 和概率矩阵 `P` 都包含 $N^2$ 个
元素。FP32 下，一个矩阵占：

$$N^2\times 4\ \text{bytes}$$

例如 `N=8192` 时，一个矩阵就需要：

$$8192^2\times 4=268{,}435{,}456\ \text{bytes}=256\ \text{MiB}$$

naive forward 同时保留 `S` 和 `P`，因此需要约 `512 MiB` workspace。
naive backward 需要 `S`、`P`、`dP`、`dS` 和长度为 `Nq` 的修正向量，
因此需要约 `1 GiB` workspace。

### 2.3 Tiled forward 为什么不需要保存完整 `S` 和 `P`

Tiled forward 每次只加载 32 个 key/value，并维护每个 query 行的 online
softmax 状态：

- $m_i$：目前见过的最大 score；
- $l_i$：以 $m_i$ 为基准的指数和；
- $a_i$：尚未除以 $l_i$ 的输出分子。

读入新 tile 后，更新规则为：

$$m_i'=\max(m_i,\max_j S_{ij}),\qquad \alpha_i=\exp(m_i-m_i')$$

$$l_i'=\alpha_i l_i+\sum_j\exp(S_{ij}-m_i')$$

$$a_i'=\alpha_i a_i+\sum_j\exp(S_{ij}-m_i')v_j$$

遍历完所有 key tile 后：

$$o_i=\frac{a_i}{l_i},\qquad L_i=m_i+\log l_i$$

因此 `S` 和 `P` 只在当前 tile 的寄存器/shared memory 生命周期内存在，
不会写成完整的 `Nq x Nk` 全局矩阵。对应实现见
[`tiled_forward.cu:L62-L136`](./src/tiled_forward.cu#L62-L136)。

### 2.4 Tiled backward 为什么分成两个 owner pass

Backward 使用以下恒等式：

$$D_i=\sum_c O_{ic}\,dO_{ic}$$

$$dP_{ij}=dO_i^\top v_j,\qquad dS_{ij}=P_{ij}(dP_{ij}-D_i)$$

$$dQ=\frac{dS K}{\sqrt d},\qquad dK=\frac{dS^\top Q}{\sqrt d},\qquad dV=P^\top dO$$

其中 $P_{ij}$ 不从 forward 保存，而是使用保存的 log-sum-exp 重新计算：

$$P_{ij}=\exp(S_{ij}-L_i)$$

不同梯度的归约方向不同：

- `dQ[i, :]` 需要固定 query 行 $i$，遍历所有 key；
- `dK[j, :]` 和 `dV[j, :]` 需要固定 key 行 $j$，遍历所有 query。

本实现因此使用三个 kernel：

1. `correction_kernel` 计算所有 $D_i$；
2. query-owner kernel 固定 query 行并完整归约 `dQ`；
3. key-owner kernel 固定 key 行并完整归约 `dK/dV`。

每个最终输出只有一个 warp 负责，所以不需要 atomic。代价是两个 owner pass
会分别重建局部 score 和 probability。实现入口见
[`tiled_backward.cu:L290-L340`](./src/tiled_backward.cu#L290-L340)，两个
owner kernel 分别见
[`tiled_backward.cu:L43-L158`](./src/tiled_backward.cu#L43-L158) 和
[`tiled_backward.cu:L160-L281`](./src/tiled_backward.cu#L160-L281)。

---

## 3. 本报告比较的四类实现

### 3.1 Naive CUDA

Naive CUDA 的目的不是追求接近 cuBLAS 的速度，而是把公式直接拆成容易审核的
kernel。

Forward：

```text
scores_kernel
    -> softmax_rows_kernel
    -> output_kernel
```

Backward：

```text
scores_kernel
    -> probabilities_from_lse_kernel
    -> grad_probabilities_kernel
    -> correction_kernel
    -> grad_scores_kernel
    -> grad_query_kernel
    -> grad_key_value_kernel
```

它显式物化 `S/P/dP/dS`。workspace 公式在
[`naive.cu:L304-L309`](./src/naive.cu#L304-L309)，launch 顺序在
[`naive.cu:L312-L407`](./src/naive.cu#L312-L407)。

### 3.2 Tiled CUDA

最终版本使用：

- 8 warps，也就是 256 threads/block；
- 一个 warp 拥有一行 query 或 key；
- 每次流式处理 32 行 key 或 query；
- shared-memory 行使用奇数 stride；
- forward 不需要全局 workspace；
- backward 只需要长度为 `Nq` 的 $D$ 向量。

这些核心常量见
[`cuda_common.cuh:L14-L24`](./src/cuda_common.cuh#L14-L24)。

### 3.3 PyTorch default 与 forced efficient

`pytorch_sdpa_default` 直接调用：

```python
F.scaled_dot_product_attention(query, key, value, is_causal=causal)
```

`pytorch_sdpa_efficient` 使用 `sdpa_kernel` 强制选择
`SDPBackend.EFFICIENT_ATTENTION`。两者的 forward/backward 时间在所有测试
shape 上几乎相同，同时 backend probe 显示 efficient 可用，因此可判断本机
default 路径实际使用了 memory-efficient backend。

需要特别注意：这不是 NVIDIA FlashAttention backend。GTX 1060 是 `sm_61`，
PyTorch 明确报告 Flash backend 只支持 `sm_80` 或更新架构。

### 3.4 PyTorch math 与显式 eager

`pytorch_sdpa_math` 强制使用 `SDPBackend.MATH`。

`pytorch_eager` 显式执行：

```python
scores = query @ key.transpose(-2, -1) / sqrt(d)
probabilities = softmax(scores)
output = probabilities @ value
```

这两条路径都允许物化二次规模张量，但底层矩阵乘法可以直接调用高度优化的
cuBLAS SGEMM。它们是“显存换吞吐”的对照组。

四个 PyTorch 路径定义在
[`pytorch_benchmark.py:L57-L102`](./app/pytorch_benchmark.py#L57-L102)。

---

## 4. 硬件和软件环境

测试日期为 2026-09-30。

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1060 |
| 显存 | 6144 MiB |
| Compute capability | 6.1 |
| Driver | 570.211.01 |
| CUDA 编译器 | 12.8.93 |
| C++ 编译目标 | `sm_61` |
| Host compiler | GCC 13.3.0 |
| PyTorch | 2.11.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| 数据类型 | FP32 |
| Batch / heads | 1 / 1 |

CUDA 12.8 仍能为 `sm_61` 生成代码，但会提示旧架构的 offline compilation
将在未来版本移除。CUDA 13 已不支持 Pascal，因此本项目不能升级到 CUDA 13
后继续编译 GTX 1060 目标。

远端目录：

```text
代码：   ~/work/cs336-fa/cuda_flash_attention
结果：   ~/var/cs336-fa/results
日志：   ~/var/cs336-fa/logs
tmux：   cs336-fa
```

代码由开发机单向 rsync 到 GPU 节点。`build/`、`build-fast/`、Python cache
和运行结果都不会同步回源码目录。

---

## 5. Benchmark 方法

### 5.1 输入

C++ 与 PyTorch 都使用：

- FP32；
- 单 batch、单 head；
- 随机值范围 `[-0.5, 0.5]`；
- 固定随机种子 `0`；
- `Q` shape 为 `(Nq, d)`；
- `K/V` shape 为 `(Nk, d)`。

C++ 使用 `std::mt19937`，PyTorch 使用自己的 CUDA RNG，因此两边不是逐元素
相同的随机输入。这里比较的是 dense attention kernel 性能，输入数值不会改变
控制流和主要访存量。两边的正确性分别在各自 reference 下验证。

### 5.2 C++ 计时

C++ benchmark 在计时前完成：

1. Host 随机输入生成；
2. `cudaMalloc`；
3. Host-to-Device 拷贝；
4. workspace 分配；
5. correctness reference 和结果回读。

计时区域只包含 kernel launch 对应的 GPU 工作：

```text
warmup launches
cudaDeviceSynchronize
CUDA Event start
measurement launches
CUDA Event stop
event synchronize
elapsed / iterations
```

实现见
[`benchmark.cu:L212-L233`](./app/benchmark.cu#L212-L233) 和
[`benchmark.cu:L315-L333`](./app/benchmark.cu#L315-L333)。

因此表中的 C++ 时间不包含：

- `cudaMalloc/cudaFree`；
- Host/Device 拷贝；
- CPU reference；
- 命令行和日志开销。

### 5.3 PyTorch 计时

PyTorch 使用相同的 CUDA Event 方法，见
[`pytorch_benchmark.py:L42-L54`](./app/pytorch_benchmark.py#L42-L54)。

Forward 在 `torch.no_grad()` 下计时。Backward 先构造一次 forward graph，
随后对同一 graph 重复调用 `torch.autograd.grad(..., retain_graph=True)`，
见
[`pytorch_benchmark.py:L127-L165`](./app/pytorch_benchmark.py#L127-L165)。

这样测到的是 GPU forward 或 backward 工作，不包含 Python wall-clock
调度时间。它与 C++ CUDA Events 的口径更接近，但仍不是完整模型端到端时间。

### 5.4 Warmup 和迭代次数

标准 `N=128..2048` sweep 使用：

| N | Warmup | Measurement |
|---:|---:|---:|
| 128、256、512 | 20 | 100 |
| 1024 | 10 | 50 |
| 2048 | 5 | 20 |

`N=4096` 使用 2 次 warmup 和 5 次 measurement，`N=8192` 使用 1 次 warmup
和 3 次 measurement。长序列数据的样本数更少，因此适合判断量级和趋势，
不应解读为严格的统计置信区间。

自动 sweep 脚本见
[`run_benchmark_suite.sh`](./scripts/run_benchmark_suite.sh)。

### 5.5 “F/B”怎么读

表格中的：

```text
3.341 / 10.720
```

表示：

- forward 平均耗时 `3.341 ms`；
- backward 平均耗时 `10.720 ms`。

加速比统一使用“对照时间除以 tiled 时间”：

$$\text{speedup}=\frac{T_{\text{baseline}}}{T_{\text{tiled}}}$$

因此：

- `speedup > 1` 表示 tiled 更快；
- `speedup < 1` 表示 tiled 更慢。

### 5.6 为什么没有把 forward 和 backward 相加

不同框架对 forward 中保存哪些 backward 中间量有不同策略。简单相加会混淆
“纯推理 forward”和“为 backward 建图的 forward”。所以本报告分别比较：

- no-grad forward；
- 已存在 forward 状态时的 backward。

真实训练端到端比较需要 PyTorch extension、相同 batch/head 布局和同一
autograd 接口，这属于后续工作。

---

## 6. 正确性与内存安全验证

### 6.1 CPU reference 有限差分

CPU reference 使用 double accumulation，并对 `Q/K/V` 分别执行中心有限差分：

$$\frac{\partial \mathcal L}{\partial x}\approx\frac{\mathcal L(x+\epsilon)-\mathcal L(x-\epsilon)}{2\epsilon}$$

测试结果：

| 模式 | `dQ` 最大绝对误差 | `dK` 最大绝对误差 | `dV` 最大绝对误差 |
|---|---:|---:|---:|
| Non-causal | `1.47e-5` | `1.06e-5` | `1.65e-5` |
| Causal | `1.98e-5` | `2.88e-5` | `2.25e-5` |

测试文件为
[`tests/reference_test.cpp`](./tests/reference_test.cpp)。

### 6.2 CUDA 与 CPU reference 对比

覆盖的 `head_dim` 包括：

```text
3, 7, 33, 64, 65, 127, 128
```

这些值刻意覆盖：

- 小于一个 warp 的维度；
- 非 32 整数倍维度；
- 跨越 32、64 边界的维度；
- tiled 实现支持的最大维度 128。

还覆盖：

- `Nq != Nk` 的矩形 attention；
- causal 和 non-causal；
- query/key 数量不是 tile 整数倍；
- 最后一个 warp 只含部分有效 lane。

检查对象包括 `O`、log-sum-exp、`dQ`、`dK`、`dV`。判定条件为：

$$|x-y|\le 5\times 10^{-3}+5\times 10^{-3}|y|$$

所有 case 通过。观察到的最大绝对误差量级不超过约 `4.77e-7`。

### 6.3 Compute Sanitizer

最终 tiled kernel 使用以下额外 case 做 CUDA 内存检查：

```text
Nq=37, Nk=53, d=64, causal=true
```

结果：

```text
========= ERROR SUMMARY: 0 errors
```

这项检查覆盖越界访问和非法设备内存访问。它不能证明所有 shape 都不存在
竞态，但结合 owner-write 设计、边界 shape 和数值对比，已经覆盖当前接口的
主要风险。

### 6.4 构建和测试状态

```text
CMake Release sm_61 build: passed
CTest: 1/1 passed
CUDA edge cases: passed
Compute Sanitizer memcheck: 0 errors
```

---

## 7. 序列长度性能结果

### 7.1 Non-causal，`head_dim=64`

| N | Naive F/B ms | Tiled F/B ms | PyTorch default F/B ms | PyTorch math F/B ms |
|---:|---:|---:|---:|---:|
| 128 | 0.076 / 0.154 | 0.034 / 0.113 | 0.038 / 0.205 | 0.132 / 0.268 |
| 256 | 0.242 / 0.505 | 0.084 / 0.265 | 0.053 / 0.660 | 0.131 / 0.263 |
| 512 | 0.782 / 1.587 | 0.224 / 0.718 | 0.087 / 2.395 | 0.126 / 0.260 |
| 1024 | 2.757 / 6.289 | 0.905 / 2.912 | 0.200 / 9.632 | 0.310 / 0.445 |
| 2048 | 10.981 / 24.865 | 3.341 / 10.720 | 0.824 / 39.957 | 1.094 / 1.483 |
| 4096 | 48.061 / 99.605 | 13.154 / 41.040 | 2.625 / 159.309 | 4.053 / 5.744 |
| 8192 | 198.337 / 419.128 | 52.373 / 165.267 | 9.326 / 637.072 | 18.047 / 23.592 |

#### 与 naive 比

Tiled 在所有测量点都更快。`N=8192` 时：

- forward：`198.337 / 52.373 = 3.79x`；
- backward：`419.128 / 165.267 = 2.54x`。

这说明即使 tiled backward 多做了一次局部 score/probability 重算，减少全局
中间矩阵读写仍足以超过当前 naive 标量实现。

#### 与 PyTorch default 比

PyTorch default forward 始终更快。`N=8192` 时：

- PyTorch default forward 为 `9.326 ms`；
- tiled forward 为 `52.373 ms`；
- tiled 慢约 `5.62x`。

但 backward 方向相反：

- PyTorch default backward 为 `637.072 ms`；
- tiled backward 为 `165.267 ms`；
- tiled 快约 `3.85x`。

这不是“自写 CUDA 全面超过 PyTorch”。准确结论是：在 GTX 1060 FP32
这个特定 backend dispatch 下，PyTorch memory-efficient backward 的表现较差，
而本实现的 owner-pass backward 更适合该硬件。

#### 与 PyTorch math 比

`N=8192` 时 PyTorch math backward 只需 `23.592 ms`，比 tiled 的
`165.267 ms` 快约 `7.00x`。原因不是 math 算法计算量更低，而是其主要计算
可以交给高度优化的 cuBLAS SGEMM；代价是约 `1 GiB` 的额外峰值显存。

### 7.2 Causal，`head_dim=64`

| N | Naive F/B ms | Tiled F/B ms | PyTorch default F/B ms | PyTorch math F/B ms |
|---:|---:|---:|---:|---:|
| 128 | 0.042 / 0.100 | 0.024 / 0.082 | 0.028 / 0.117 | 0.174 / 0.248 |
| 256 | 0.121 / 0.319 | 0.054 / 0.187 | 0.049 / 0.378 | 0.177 / 0.240 |
| 512 | 0.439 / 1.335 | 0.170 / 0.573 | 0.089 / 1.358 | 0.178 / 0.261 |
| 1024 | 1.745 / 5.329 | 0.617 / 2.189 | 0.174 / 5.184 | 0.479 / 0.446 |
| 2048 | 6.884 / 20.666 | 2.243 / 8.129 | 0.439 / 20.284 | 1.739 / 1.480 |
| 4096 | 27.962 / 83.389 | 9.131 / 31.244 | 1.421 / 82.030 | 6.715 / 5.705 |
| 8192 | 125.493 / 348.362 | 37.080 / 126.566 | 4.649 / 326.708 | 28.425 / 23.064 |

`N=8192` 时：

- tiled 相对 naive 为 `3.38x` forward 和 `2.75x` backward；
- tiled backward 相对 PyTorch default 快 `2.58x`；
- tiled backward 相对 PyTorch math 慢 `5.49x`。

### 7.3 为什么 causal 没有严格快一倍

理论上 causal attention 只保留下三角区域，有效 query-key pair 接近
non-causal 的一半。但实际运行仍包含：

- tile 加载和 block 同步；
- 对角 tile 内的逐 lane mask；
- online softmax 状态更新；
- kernel launch；
- 边界 tile；
- key-owner backward 对完全不可见 query tile 的循环控制。

当前 query-owner pass 可以跳过完整的未来 key tile，但 key-owner pass 还没有
对完整不可见的 query tile 做同等级的 tile-level early skip。因此 causal
通常比 non-causal 快，但不会稳定达到理想的 `2x`。

### 7.4 小 shape 为什么不能过度解读

`N=128` 时，tiled non-causal forward 为 `0.034 ms`，PyTorch default 为
`0.038 ms`。这个差异只有约 `4 us`，容易受到以下因素影响：

- GPU 从 P8 升到 P0 的频率变化；
- CUDA launch 和 Event 分辨率；
- 第一个 case 的 cache 和上下文状态；
- 不同实现的 kernel 数量。

因此报告只把 `N>=512` 的趋势作为主要结论，不宣称自写 forward 在小 shape
上稳定超过 PyTorch。

---

## 8. Head dimension 扫描

固定 `N=1024`，比较 `d=32/64/128`。

| d | 模式 | Naive F/B ms | Tiled F/B ms | PyTorch default F/B ms | PyTorch math F/B ms |
|---:|---|---:|---:|---:|---:|
| 32 | Non-causal | 1.928 / 4.100 | 0.506 / 1.567 | 0.187 / 9.211 | 0.274 / 0.360 |
| 64 | Non-causal | 2.757 / 6.289 | 0.905 / 2.912 | 0.200 / 9.632 | 0.310 / 0.445 |
| 128 | Non-causal | 5.372 / 12.224 | 2.688 / 7.950 | 0.365 / 15.299 | 0.397 / 0.603 |
| 32 | Causal | 0.965 / 2.886 | 0.309 / 1.044 | 0.157 / 4.941 | 0.442 / 0.358 |
| 64 | Causal | 1.745 / 5.329 | 0.617 / 2.189 | 0.174 / 5.184 | 0.479 / 0.446 |
| 128 | Causal | 3.369 / 10.117 | 1.805 / 5.976 | 0.199 / 7.978 | 0.564 / 0.602 |

可以得到三个结论。

第一，tiled 在三个 head dimension 上都快于 naive，因此收益不局限于
`d=64`。

第二，`d` 增大时，tiled 的时间增长明显。例如 non-causal forward 从
`d=32` 的 `0.506 ms` 增长到 `d=128` 的 `2.688 ms`，超过简单的四倍关系。
原因包括：

- 每个 lane 的 score 点积是串行 feature 循环；
- 每个线程需要更多 output/gradient accumulator；
- dynamic shared memory 增长；
- `d=128` 时一个 block 使用约 36.3 到 40.3 KiB shared memory；
- forward kernel 使用 64 registers/thread。

第三，PyTorch default backward 也随 `d` 增大，但在这些 shape 上仍慢于
tiled。PyTorch math 则继续依赖 SGEMM 获得明显更高的吞吐。

---

## 9. 显存分析

### 9.1 C++ workspace 的精确定义

C++ 表中的 workspace 只表示算法要求调用方额外提供的临时设备内存，不包括：

- 输入 `Q/K/V`；
- 输出 `O/L`；
- 上游梯度 `dO`；
- 最终梯度 `dQ/dK/dV`。

Naive forward：

$$M_{\text{naive,fwd}}=2N_qN_k\times 4\ \text{bytes}$$

Naive backward：

$$M_{\text{naive,bwd}}=(4N_qN_k+N_q)\times 4\ \text{bytes}$$

Tiled forward：

$$M_{\text{tiled,fwd}}=0$$

Tiled backward：

$$M_{\text{tiled,bwd}}=N_q\times 4\ \text{bytes}$$

### 9.2 实测规模

| N | Naive forward workspace | Naive backward workspace | Tiled backward workspace |
|---:|---:|---:|---:|
| 512 | 2.000 MiB | 4.002 MiB | 2 KiB |
| 2048 | 32.000 MiB | 64.008 MiB | 8 KiB |
| 4096 | 128.000 MiB | 256.016 MiB | 16 KiB |
| 8192 | 512.000 MiB | 1024.031 MiB | 32 KiB |

在 `N=8192` 时，naive backward workspace 与 tiled backward workspace 的
比值约为：

$$\frac{1{,}073{,}774{,}592}{32{,}768}=32{,}769$$

也就是 tiled 的额外 workspace 小约 32769 倍。

### 9.3 PyTorch 显存数据为什么不能直接等同于 C++ workspace

PyTorch benchmark 在清理 cache 和重置峰值统计后，记录从克隆
`Q/K/V`、执行 forward 到完成一次 backward 的
`max_memory_allocated - baseline`。实现见
[`pytorch_benchmark.py:L105-L124`](./app/pytorch_benchmark.py#L105-L124)。

这个值包含：

- 克隆后的输入；
- output；
- autograd 保存状态；
- `dQ/dK/dV`；
- backend 内部临时量。

所以它比 C++ 的“纯 workspace”口径更宽，不能直接做常数倍比较。但可以比较
增长趋势：

| N | PyTorch default 额外峰值 | PyTorch math 额外峰值 |
|---:|---:|---:|
| 512 | 1.004 MiB | 4.875 MiB |
| 2048 | 4.016 MiB | 67.500 MiB |
| 4096 | 8.031 MiB | 263.000 MiB |
| 8192 | 16.062 MiB | 1038.000 MiB |

PyTorch default 近似线性增长，说明 memory-efficient backend 确实避免了完整
二次矩阵；PyTorch math 近似二次增长，与完整 attention 中间量一致。

### 9.4 显存与速度的真正权衡

`N=8192` non-causal backward：

| 实现 | 时间 | 临时内存特征 |
|---|---:|---|
| Tiled CUDA | 165.267 ms | 32 KiB C++ workspace |
| PyTorch default | 637.072 ms | 16.062 MiB 额外峰值，近似线性 |
| PyTorch math | 23.592 ms | 1038 MiB 额外峰值，二次增长 |

这组数据体现的不是“某个实现绝对最好”，而是三种不同选择：

- tiled：极小 workspace，速度中等；
- PyTorch default：同样内存高效，forward 很快，但 Pascal backward 很慢；
- PyTorch math：最快，但显存随 $N^2$ 增长。

---

## 10. 三项关键优化为什么有效

### 10.1 优化前后总结果

固定 `N=2048,d=64`：

| 版本 | Non-causal F/B ms | Causal F/B ms |
|---|---:|---:|
| 初始 4-warps、未 padding | 21.248 / 76.080 | 12.199 / 41.357 |
| 8-warps、未 padding | 19.035 / 72.711 | 10.398 / 37.418 |
| 8-warps、padding | 3.602 / 11.779 | 2.282 / 9.020 |
| 最终版，跳过无效 slot | 3.341 / 10.720 | 2.243 / 8.129 |

初始版到最终版：

| 模式 | Forward 加速 | Backward 加速 |
|---|---:|---:|
| Non-causal | `6.36x` | `7.10x` |
| Causal | `5.44x` | `5.09x` |

不同版本在独立进程中测量，微小差异会受到 GPU 动态频率影响；但 padding
带来的 5 倍以上变化远大于测量噪声。

### 10.2 从 4 warps/block 增加到 8 warps/block

初始版本一个 block 只有 4 个 owner warp：

```text
block
├── warp 0 -> query/key row 0
├── warp 1 -> query/key row 1
├── warp 2 -> query/key row 2
└── warp 3 -> query/key row 3
```

最终版本使用 8 个 warp，也就是 256 threads/block。收益包括：

- 同一份 streamed tile 被更多 owner row 复用；
- shared-memory 容量允许时，每个 block 暴露更多可调度 warp；
- `d=128` 时即使只能驻留一个大 shared-memory block，也从 4 个 warp
  增加到 8 个 warp。

它带来了约 5% 到 30% 的收益，但不是最大改进，因为原实现还存在严重的
shared-memory bank conflict。

### 10.3 Shared-memory bank conflict

GTX 1060 的 shared memory 有 32 个 bank。FP32 每个元素占一个 32-bit word，
可以用下面的简化关系理解 bank：

$$\operatorname{bank}(\text{word index})=\text{word index}\bmod 32$$

原始 row-major shared-memory 下标为：

$$\text{index}=\text{lane}\times d+\text{feature}$$

当 `d=64` 时：

$$\operatorname{bank}=(64\times\text{lane}+\text{feature})\bmod 32=\text{feature}\bmod 32$$

一个 warp 的 32 个 lane 会访问同一个 bank 中的 32 个不同地址，形成严重
bank conflict，访问被拆成多次处理。`d=128` 同样如此。

最终版本把 shared-memory 行 stride 改成奇数：

```cpp
shared_stride = head_dim + (head_dim is even ? 1 : 0)
```

`d=64` 时 stride 变为 65：

$$\operatorname{bank}=(65\times\text{lane}+\text{feature})\bmod 32=(\text{lane}+\text{feature})\bmod 32$$

此时 32 个 lane 映射到 32 个不同 bank，冲突消失。实现见
[`cuda_common.cuh:L22-L24`](./src/cuda_common.cuh#L22-L24)，具体 shared-memory
下标见
[`tiled_forward.cu:L36-L72`](./src/tiled_forward.cu#L36-L72) 和
[`tiled_backward.cu:L58-L99`](./src/tiled_backward.cu#L58-L99)。

这是本轮最大的性能改进，说明 tiled 算法“减少 HBM 流量”并不自动等于高性能；
片上 shared-memory 的布局同样决定实际吞吐。

### 10.4 跳过无效 feature slot

为支持 `head_dim <= 128`，每个 lane 最多维护 4 个 feature slot：

```text
slot 0 -> feature lane + 0
slot 1 -> feature lane + 32
slot 2 -> feature lane + 64
slot 3 -> feature lane + 96
```

在 `d=64` 时只有前两个 slot 有效。初始修正版为了保证所有 lane 都参与
full-mask `__shfl_sync`，仍然执行了 slot 2 和 slot 3 的 shuffle，只是不累加
结果。

最终版本在 slot 外层加入 warp-uniform 条件：

```cpp
if (feature_base < head_dim) {
    // 所有 lane 一起执行这一 slot 的 shuffle
}
```

因为 `feature_base` 和 `head_dim` 对整个 warp 相同，这个分支不会造成
shuffle participation 错误。`d=64` 因此省掉一半无效 slot 的 shuffle。
实现见
[`tiled_forward.cu:L96-L116`](./src/tiled_forward.cu#L96-L116)、
[`tiled_backward.cu:L123-L142`](./src/tiled_backward.cu#L123-L142) 和
[`tiled_backward.cu:L239-L264`](./src/tiled_backward.cu#L239-L264)。

### 10.5 为什么没有默认启用 fast math

CMake 提供 `FA_USE_FAST_MATH=ON`，它会向 CUDA 编译器传入
`--use_fast_math`。矩形 causal/non-causal correctness case 都通过。

在 GPU 已升频后的交替 A/B 中，`N=2048,d=64` non-causal 结果为：

| 构建 | Forward ms | Backward ms |
|---|---:|---:|
| Default，第 2 次稳定测量 | 3.389 | 10.757 |
| Fast math，第 2 次稳定测量 | 3.351 | 10.787 |

forward 差异约 1%，backward 没有收益，处于动态频率和测量波动范围内。因此：

- 保留 fast-math 作为实验开关；
- 默认关闭，避免无收益地改变除法、平方根等数值语义；
- 不把第一次冷状态下的较大差异当作可靠加速。

---

## 11. Nsight Systems 结果

对最终版本的 `N=2048,d=64` non-causal tiled 路径采集 Nsight Systems。
Profiler 会引入额外开销，因此这里只看 kernel 时间占比，不把 profile 下的
绝对毫秒数与普通 benchmark 混用。

| Kernel | GPU 总时间占比 | 作用 |
|---|---:|---|
| `tiled_grad_key_value_kernel` | 40.4% | 固定 key 行，沿 query 归约 `dK/dV` |
| `tiled_grad_query_kernel` | 35.8% | 固定 query 行，沿 key 归约 `dQ` |
| `tiled_forward_kernel` | 23.7% | online softmax 与输出累计 |
| `correction_kernel` | 0.1% | 计算 $D_i=\langle O_i,dO_i\rangle$ |

每个主 kernel 共观察到 5 次实例，对应一次预执行、一次 warmup 和三次正式
measurement 中的相关调用。

### 11.1 这张表说明什么

Backward 的两个主 pass 合计占约 76.2%，是下一轮优化重点。它们都包含：

- 重新计算 score；
- 从 log-sum-exp 重建 probability；
- 计算 `dP/dS`；
- 对 feature 维做串行点积；
- 用 warp shuffle 将每个 pair 的标量贡献广播给 feature owner。

`correction_kernel` 只有 0.1%。即使把它完全消除，整体收益也不会超过约
0.1%，因此当前不值得优先融合。

### 11.2 资源使用

`cuobjdump --dump-resource-usage` 给出的寄存器数量：

| Kernel | Registers/thread |
|---|---:|
| `tiled_forward_kernel` | 64 |
| `tiled_grad_query_kernel` | 48 |
| `tiled_grad_key_value_kernel` | 32 |
| `correction_kernel` | 30 |

`head_dim=128` 时动态 shared memory：

| Kernel 类别 | Dynamic shared memory/block |
|---|---:|
| Forward | 约 36.3 KiB |
| Backward owner kernel | 约 40.3 KiB |

GTX 1060 每 block 的常规上限是 48 KiB，所以 shape 可以启动；但大 shared
memory 和 forward 的 64 registers/thread 会限制 occupancy。这也是不能继续
盲目增加 warps 或 tile 尺寸的原因。

---

## 12. 如何解释“我们的实现是否有优势”

答案取决于对照对象和资源约束。

### 12.1 相对 naive CUDA

优势明确：

- forward 和 backward 都更快；
- workspace 从二次规模降到线性或零；
- 长序列越能体现差距；
- causal 与 non-causal 都成立；
- `d=32/64/128` 都成立。

这是当前实现已经证明的主要算法价值。

### 12.2 相对 PyTorch default SDPA

优势与劣势分开看：

| 方向 | 结果 |
|---|---|
| Forward | PyTorch default 明显更快，尤其长序列 |
| Backward | 本实现明显更快，`N=8192` 快 `2.58x-3.85x` |
| 显存增长 | 两者都避免完整二次矩阵 |
| 工程能力 | PyTorch 支持 batch/head、dtype、autograd 和广泛 shape；本实现目前不支持 |

因此不能说“整体超过 PyTorch”。更准确的说法是：本实现展示了一个
Pascal-oriented backward 调度，在本机 FP32 条件下优于 PyTorch 默认选择的
memory-efficient backward，但 forward 和通用性仍有明显差距。

### 12.3 相对 PyTorch math

PyTorch math 在当前所有可容纳 shape 上通常最快，因为核心工作由 cuBLAS
SGEMM 完成。但它会物化二次规模中间量。

如果显存足够、目标只是 GTX 1060 上的最快运行时间，PyTorch math 是当前
更实用的选择。如果序列继续增长导致二次张量不可接受，tiled 或 PyTorch
memory-efficient 才体现其必要性。

### 12.4 按场景选择

| 使用场景 | 推荐 |
|---|---|
| 学习 online softmax、owner pass 和 CUDA tiling | 本项目 tiled 实现 |
| 验证公式和 CUDA kernel correctness | CPU reference + naive + tiled |
| GTX 1060、显存充足、追求最快 backward | PyTorch math |
| GTX 1060、显存受限、forward-only | PyTorch default |
| GTX 1060、显存受限、需要 backward | 对比 tiled 与 PyTorch default；当前数据倾向 tiled |
| 现代 `sm_80+` GPU | 应重新对比官方 Flash SDPA，不能沿用本报告结论 |
| 实际模型训练 | 先补 batch/head 和 PyTorch extension，再做端到端测试 |

---

## 13. 后续优化路线

### 13.1 P0：按 `head_dim` 模板化专用 kernel

当前 `head_dim` 是运行时参数，点积循环也是运行时循环。可为常见值建立：

```text
head_dim = 32
head_dim = 64
head_dim = 128
```

然后模板 dispatch。预期收益：

- 编译器可完整展开 feature 循环；
- 减少动态边界判断和整数地址计算；
- 更容易使用 `float2/float4`；
- 可为不同 `d` 选择不同 warps/block 和 shared-memory 布局。

风险是代码体积和维护成本增加，因此只应特化常见维度，保留 generic fallback。

### 13.2 P0：重写点积 microkernel

当前一个 lane 负责一个 query-key pair 的完整 feature 点积，feature 方向串行：

```text
lane 0: dot(q, k0)
lane 1: dot(q, k1)
...
lane 31: dot(q, k31)
```

这个设计容易理解，但 FMA 指令级并行度不足，也没有形成接近 SGEMM 的
register tile。可尝试：

- 每 lane 同时维护多个 score accumulator；
- 每 warp 同时处理多个 query；
- Q/K 使用 `float4` 向量读取；
- 对 shared-memory tile 做适合 warp 消费的转置；
- 将加载、FMA、shuffle 的执行重叠起来。

这是缩小与 PyTorch/cuBLAS 差距的核心工作。

### 13.3 P1：分别调优三个主 kernel

当前 forward、`dQ`、`dK/dV` 共用 `kWarpsPerBlock=8`，但三者资源特征不同：

- forward 为 64 registers/thread；
- `dQ` 为 48 registers/thread；
- `dK/dV` 为 32 registers/thread；
- shared-memory 组成也不同。

更合理的方案是分别配置：

```text
forward warps/block
dQ warps/block
dK/dV warps/block
```

并对 `d=32/64/128` 做小规模搜索。不能假设同一个 tile shape 对三个 kernel
都最优。

### 13.4 P1：完善 causal tile-level skip

Query-owner 已能跳过 query 右侧完全不可见的 key tile。Key-owner 还可以增加：

```text
如果当前 query tile 的最大 query index < key index：
    整个 tile 不计算 score、probability 和梯度贡献
```

这会减少 causal `dK/dV` 中全零 tile 的 shuffle 和循环工作，使 causal
backward 更接近理论上的半矩阵成本。

### 13.5 P1：评估一次重算同时服务三个梯度

当前两个 owner pass 分别重建 `P/dS`：

- query-owner 为 `dQ` 重建一次；
- key-owner 为 `dK/dV` 再重建一次。

可选方案是在 key-owner pass 中同时生成 `dQ` partial：

- atomic add 到 `dQ`；
- 或写入 partial buffer，最后再归约。

这会减少重算，但引入：

- atomic 冲突或非确定性；
- 额外 partial buffer；
- 额外 reduction kernel；
- 更复杂的数据所有权。

是否值得必须由 benchmark 决定，不能只根据 FLOPs 推断。

### 13.6 P2：PyTorch extension 与批量维度

当前 C++ benchmark 是单 head 连续矩阵。要评估真实训练收益，需要：

1. 增加 batch/head 外层 dispatch；
2. 支持 PyTorch tensor 和当前 stream；
3. 编写 autograd wrapper；
4. 与模型中的 layout、transpose、contiguous 成本一起计时；
5. 检查多 head 下 block 数量是否足以占满所有 SM。

否则当前数据只能说明单 head kernel 性能，不能直接换算整个 Transformer 的
端到端加速。

### 13.7 P2：增强统计稳定性

当前工具输出平均值。后续正式报告可增加：

- 每次 iteration 的原始样本；
- median、P10、P90、标准差；
- GPU temperature、P-state 和时钟；
- 随机化 implementation 执行顺序；
- 锁定 GPU 时钟后的复测；
- Nsight Compute 的 bank conflict、occupancy、memory throughput 和
  instruction throughput 指标。

这能区分真实优化与动态频率噪声。

### 13.8 当前不应优先做的工作

| 项目 | 原因 |
|---|---|
| 融合 `correction_kernel` | 仅占 GPU 时间约 0.1% |
| 默认启用 fast math | A/B 没有稳定收益 |
| 继续盲目增加 warps/block | 已受 shared memory 和寄存器约束 |
| 在 GTX 1060 上模仿 Tensor Core kernel | 硬件不支持 |
| 使用 `cp.async` | Pascal 不支持 |

---

## 14. 复现实验

### 14.1 构建

```bash
cmake -S cuda_flash_attention -B cuda_flash_attention/build \
  -DFA_ENABLE_CUDA=ON \
  -DFA_CUDA_ARCHITECTURE=61 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cuda_flash_attention/build -j
```

可选 fast-math 实验：

```bash
cmake -S cuda_flash_attention -B cuda_flash_attention/build-fast \
  -DFA_ENABLE_CUDA=ON \
  -DFA_CUDA_ARCHITECTURE=61 \
  -DFA_USE_FAST_MATH=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cuda_flash_attention/build-fast -j
```

### 14.2 正确性

```bash
ctest --test-dir cuda_flash_attention/build --output-on-failure

cuda_flash_attention/build/fa_benchmark \
  --nq 257 \
  --nk 193 \
  --head-dim 64 \
  --causal \
  --impl both \
  --warmup 2 \
  --iterations 5
```

### 14.3 标准性能 sweep

```bash
FA_PYTHON="$HOME/.venvs/cs336-profile-cu126/bin/python" \
  cuda_flash_attention/scripts/run_benchmark_suite.sh \
  "$HOME/var/cs336-fa/results/reproduction"
```

只重跑 C++：

```bash
FA_SKIP_PYTORCH=1 \
  cuda_flash_attention/scripts/run_benchmark_suite.sh \
  "$HOME/var/cs336-fa/results/cpp-only"
```

### 14.4 汇总结果

```bash
python cuda_flash_attention/scripts/summarize_benchmarks.py \
  --cpp-dir "$HOME/var/cs336-fa/results/final" \
  --pytorch-dir "$HOME/var/cs336-fa/results/baseline" \
  --before-cpp-dir "$HOME/var/cs336-fa/results/baseline" \
  --output "$HOME/var/cs336-fa/results/summary.md"
```

### 14.5 Nsight Systems

```bash
nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="$HOME/var/cs336-fa/results/tiled_n2048" \
  cuda_flash_attention/build/fa_benchmark \
    --nq 2048 \
    --nk 2048 \
    --head-dim 64 \
    --impl tiled \
    --warmup 1 \
    --iterations 3 \
    --no-verify

nsys stats \
  --report cuda_gpu_kern_sum \
  "$HOME/var/cs336-fa/results/tiled_n2048.nsys-rep"
```

---

## 15. 局限性

阅读结果时必须保留以下边界：

1. 只有一张 GTX 1060，结论不能外推到 A100、H100、B200 或其它现代 GPU。
2. 只测试 FP32；FP16/BF16 的 backend dispatch 和吞吐关系可能完全不同。
3. 只测试单 batch、单 head；真实模型会改变并行度和 launch amortization。
4. C++ 与 PyTorch 没有通过同一个 extension 接口调用，比较的是 GPU kernel
   时间，不是完整框架端到端延迟。
5. C++ workspace 与 PyTorch peak allocated 的口径不同，只能比较增长趋势。
6. `N=4096/8192` 的 measurement 次数较少，适合看量级，不适合报告微小差异。
7. GPU 使用动态频率，首次运行可能处于 P8；微秒级小 shape 数据容易受影响。
8. 本实现追求可读性，没有生产实现中的大量 shape/dtype/template dispatch。

---

## 16. 最终结论

本实验已经验证了 FlashAttention 的核心工程命题：

1. 不物化完整 `S/P/dP/dS`，确实能把额外 workspace 从二次规模降到线性规模；
2. 正确的 tile ownership 可以在不使用 atomic 的情况下完成 `dQ/dK/dV`；
3. tiled 算法本身不保证快，shared-memory bank conflict 足以吞掉绝大部分收益；
4. 消除 bank conflict 后，当前实现相对 naive 获得了稳定的 forward/backward
   加速；
5. 与生产框架比较时必须同时看 backend、方向和显存：PyTorch default
   forward 更强，本实现 backward 在 Pascal 上更强，PyTorch math 在显存允许
   时吞吐最高；
6. 下一阶段应投入到矩阵乘风格的 register microkernel 和按 head dimension
   专用化，而不是继续优化只占 0.1% 的辅助 kernel。
