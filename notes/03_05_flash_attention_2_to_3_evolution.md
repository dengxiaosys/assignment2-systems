# 从 FlashAttention-2 到 FlashAttention-3：Hopper 异步流水线与 FP8

## 0. 本文要回答什么

FlashAttention-2（FA2）已经解决了两个核心问题：

1. 不在 HBM 中物化完整的 attention score 和 probability 矩阵；
2. 通过更好的 thread block 与 warp 工作划分，让 exact attention 更接近 GEMM 的效率。

但同一套实现直接运行在 Hopper H100 上时，仍没有充分利用 Hopper 新增的异步执行能力。FlashAttention-3（FA3）的核心不是改变 attention 数学，而是重新安排以下三类工作何时发生、由谁发起：

1. HBM 与 shared memory 之间的数据搬运；
2. Tensor Core 上的两个块矩阵乘；
3. CUDA core 与特殊函数单元上的 max、exp、求和、缩放和类型转换。

本文面向已经读过 [03_02 FA2 Forward](./03_02_flash_attention_2_beginner_textbook.md) 和 [03_03 FA2 Backward](./03_03_flash_attention_2_backward.md) 的读者，但不依赖那两篇才能阅读。本文会重新给出理解 FA3 所需的最小数学和 GPU 背景。

读完后，应当能够回答：

- 为什么 FA2 在 A100 上很成功，在 H100 上却只达到约 35% 的利用率；
- TMA、WGMMA、warpgroup 和 warp specialization 分别解决什么问题；
- inter-warpgroup ping-pong 与 intra-warpgroup 两阶段流水线有何不同；
- 为什么 softmax 的 FLOPs 很少，却可能消耗接近 GEMM 一半的时间；
- 为什么 FP8 attention 不能只把 `dtype` 从 FP16 改成 FP8；
- FA3 论文实际覆盖了哪些 forward/backward、dtype 和场景，哪些仍未证明；
- CUDA/CUTLASS 或 Triton 实现需要显式处理哪些布局、同步和资源权衡。

### 0.1 证据标签

本文刻意区分三类陈述：

- **论文事实**：论文直接陈述、算法直接给出或实验直接测得的结论；
- **推导**：从论文公式、硬件峰值或算法循环直接计算出的结果；
- **工程解释**：帮助建立直觉或指导实现的解释，不冒充论文原文或普遍性能保证。

若某项“支持”只出现在论文算法或 benchmark 中，本文只描述论文证据，不把它外推成当前某个软件版本的完整 API 支持矩阵。

### 0.2 主要一手资料

| 简称 | 本地论文 | arXiv 官方页面 | 本文主要使用范围 |
|---|---|---|---|
| FA2 | [FlashAttention-2 PDF](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf) | [arXiv:2307.08691v1](https://arxiv.org/abs/2307.08691)、[HTML](https://arxiv.org/html/2307.08691v1) | §2-§5、Algorithm 1-2、Figure 2-7、Table 1 |
| FA3 | [FlashAttention-3 PDF](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf) | [arXiv:2407.08608v2](https://arxiv.org/abs/2407.08608)、[HTML](https://arxiv.org/html/2407.08608v2) | §2-§5、Algorithm 1-2、Figure 1-7、Table 1-3、Appendix B-C |
| Hopper | 本文仍以 FA3 §2.2 为主 | [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA、shared memory、register 与异步 copy 的硬件语义 |

后文的“页码”均指本地 PDF 显示的论文页码。

---

## 1. 先看结论：FA3 相对 FA2 到底变了什么

**论文事实：** FA2 的三个主要改进是减少 non-matmul FLOPs、沿 query 序列维增加 thread blocks，以及在 CTA 内从 sliced-K 改成 sliced-Q。FA3 继承这些设计，再增加三项 Hopper 导向的改进：producer-consumer warp specialization、GEMM 与 softmax 的异步重叠、FP8 block quantization 与 incoherent processing。[FA2 §3](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf#page=5)；[FA3 §1、§3](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=1)

| 维度 | FA2 | FA3 |
|---|---|---|
| 数学结果 | tiled exact attention，浮点舍入次序可能不同 | FP16/BF16 路径保持同一数学目标；FP8 路径额外引入量化误差 |
| 主要目标硬件 | 论文围绕 A100 优化 | 论文围绕 Hopper H100 优化 |
| CTA 级并行 | batch、head、query tiles | 继承 FA2 的 CTA 网格 |
| CTA 内工作划分 | sliced-Q，减少 warp 间归约 | producer warpgroup、consumer warpgroups，backward 另有 `dQ` writer |
| 数据搬运 | 算法描述为显式 HBM -> SRAM load | producer 用 TMA 异步填充多阶段 circular SMEM buffer |
| Tensor Core 接口 | FA2 论文未使用 Hopper 专属新指令 | consumer 用异步 WGMMA |
| softmax 调度 | 与两次 GEMM 基本串行地推进 tile | 跨 consumer warpgroups ping-pong，并在单个 warpgroup 内跨迭代流水 |
| 低精度 | 论文主要讨论 FP16/BF16 | 增加 FP8 forward、块量化、不相干处理和布局变换 |
| 渐近复杂度 | $O(N^2d)$ 算术，不物化 $N^2$ 中间量 | 与 FA2 相同，主要改变常数、重叠和硬件利用率 |

**工程解释：** FA2 主要回答“怎样少搬数据、让更多 CTA/warps 做有效工作”；FA3 进一步回答“当搬运、Tensor Core 和 softmax 由不同硬件执行时，怎样让它们同时忙”。因此 FA3 是调度和数据表示的升级，不是新的 attention 定义。

---

## 2. 不变的数学骨架

### 2.1 列向量约定与框架张量布局

本文中单个 query、key、value 都是列向量：$q_i,k_j,v_j\in\mathbb{R}^d$。单个 score 与输出为：

$$s_{ij}=\alpha q_i^\top k_j,\qquad p_{ij}=\frac{\exp(s_{ij})}{\sum_{u=1}^{N_k}\exp(s_{iu})},\qquad o_i=\sum_{j=1}^{N_k}p_{ij}v_j,\qquad \alpha=\frac{1}{\sqrt d}$$

为了批量计算，把列向量的转置堆成矩阵的行：

$$Q=\begin{bmatrix}q_1^\top\\ \cdots\\ q_{N_q}^\top\end{bmatrix},\quad K=\begin{bmatrix}k_1^\top\\ \cdots\\ k_{N_k}^\top\end{bmatrix},\quad V=\begin{bmatrix}v_1^\top\\ \cdots\\ v_{N_k}^\top\end{bmatrix},\quad S=\alpha QK^\top,\quad P=\operatorname{rowsoftmax}(S),\quad O=PV$$

数学上以列向量记 $y=Wx$；在 PyTorch 中特征位于最后一维，对应实现为 `y = x @ W.T`。同理，PyTorch 中常见的 `Q @ K.transpose(-2, -1)` 是把每个 $q_i^\top$ 作为 Tensor 的一行存储，并没有把数学中的 $q_i$ 改成行向量。

**论文事实：** FA2 Algorithm 1 与 FA3 Algorithm 1-2 都计算同一个 $O=\operatorname{softmax}(\alpha QK^\top)V$，并都以 query tile 为 CTA 的输出所有权单位。[FA2 Algorithm 1](https://arxiv.org/html/2307.08691v1#S3.SS1.SSS1)；[FA3 §3.1](https://arxiv.org/html/2407.08608v2#S3.SS1)

### 2.2 FA2 留给 FA3 的 online-softmax 状态

固定一个 query tile $Q_i\in\mathbb{R}^{B_r\times d}$。第 $j$ 个 key/value tile 产生：

$$S_i^{(j)}=\alpha Q_iK_j^\top\in\mathbb{R}^{B_r\times B_c}$$

FA2 对每行维护 running maximum $m_i$、指数和 $\ell_i$ 与未归一化输出 accumulator $\widetilde O_i$。处理新 tile 时，令 $m_i^{\mathrm{new}}$ 为旧最大值与新 score 行最大值的逐行最大值，则：

$$\widetilde P_i^{(j)}=\exp\left(S_i^{(j)}-m_i^{\mathrm{new}}\right),\qquad \ell_i^{\mathrm{new}}=\exp\left(m_i^{\mathrm{old}}-m_i^{\mathrm{new}}\right)\ell_i^{\mathrm{old}}+\operatorname{rowsum}\left(\widetilde P_i^{(j)}\right)$$

$$\widetilde O_i^{\mathrm{new}}=\operatorname{diag}\left(\exp\left(m_i^{\mathrm{old}}-m_i^{\mathrm{new}}\right)\right)\widetilde O_i^{\mathrm{old}}+\widetilde P_i^{(j)}V_j$$

循环结束后才归一化并保存 log-sum-exp：

$$O_i=\operatorname{diag}(\ell_i)^{-1}\widetilde O_i,\qquad L_i=m_i+\log\ell_i$$

**论文事实：** 这是 FA2 为减少 non-matmul FLOPs 所做的关键改写。它只在最后归一化输出，并只为 backward 保存 $L$，不保存完整 $S$ 或 $P$。[FA2 §3.1.1、Algorithm 1，PDF pp.5-7](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf#page=5)

**工程解释：** FA3 没有替换这套不变量，而是拆开、延后并交错其中的步骤。理解 FA3 时，应持续追踪四个对象：当前 score tile、当前未归一化 probability tile、running softmax 状态 $(m,\ell)$、输出 accumulator $\widetilde O$。

### 2.3 Backward 的不变接口

FA2/FA3 backward 都可以从 $Q,K,V,O,dO,L$ 重算当前 tile 的 $S,P$。这里用 $r$ 表示 query 行、$c$ 表示 feature 坐标，并定义每行标量：

$$D_r=\sum_{c=1}^{d}O_{rc}dO_{rc}$$

令 $D_i$ 收集 query tile $i$ 内各行的 $D_r$，则当前 tile 中有：

$$P_i^{(j)}=\exp\left(S_i^{(j)}-L_i\right),\qquad dP_i^{(j)}=dO_iV_j^\top,\qquad dS_i^{(j)}=P_i^{(j)}\circ\left(dP_i^{(j)}-D_i[:,None]\right)$$

随后累加 $dV_j\mathrel{+}=P_i^{(j)\top}dO_i$、$dK_j\mathrel{+}=\alpha dS_i^{(j)\top}Q_i$ 和 $dQ_i\mathrel{+}=\alpha dS_i^{(j)}K_j$。

**论文事实：** FA2 Algorithm 2 给出上述重算；FA3 Appendix B.1 的 Algorithm 3 保留相同数学依赖，只改变数据搬运、WGMMA 发起与 `dQ` 写回的角色划分。[FA2 Algorithm 2，PDF p.7](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf#page=7)；[FA3 Algorithm 3，PDF p.18](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=18)

---

## 3. 理解 FA3 所需的 Hopper 背景

### 3.1 从 thread 到 warpgroup

Hopper 上与本文相关的执行层级为：

```text
thread
  -> warp: 32 threads
    -> warpgroup: 4 contiguous warps = 128 threads
      -> CTA / thread block: 同驻一个 SM，共享该 CTA 的 SMEM
        -> thread block cluster
          -> grid
```

**论文事实：** FA3 §2.2 明确定义一个 warpgroup 为 4 个连续 warps；同一 CTA 的线程共同访问 shared memory，而寄存器由线程私有。H100 SXM5 的论文配置有 80 GiB HBM、50 MiB L2、每 SM 228 KiB shared memory 和每 SM 256 KiB register file。[FA3 §2.2、Table 1，PDF p.3](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=3)

**工程解释：** warpgroup 不是“更大的 warp”。普通 warp 指令仍以 32 个线程为执行组；WGMMA 则要求 4 个连续 warps 协作发起和消费一个 warpgroup 级矩阵乘。FA3 以 warpgroup 为单位分配 producer/consumer 角色，是因为 WGMMA、寄存器预算和调度都在这个粒度上耦合。

### 3.2 TMA：让数据搬运成为独立流水线

TMA 是 Tensor Memory Accelerator。对 FA3 最重要的能力是：

- 由专用硬件异步执行 GMEM/HBM 与 SMEM 之间的张量搬运；
- 一个线程可以发起大块多维数据搬运，不需要整组线程逐元素执行 load/store 地址计算；
- copy 在途时，CTA 中其他线程可以继续计算；
- 通过 barrier 或 pipeline 状态判断某个 SMEM stage 何时“已填满”或“已消费”。

**论文事实：** FA3 Algorithm 1 的 producer 发起 $Q_i$、$K_j$、$V_j$ 的 TMA load，并用 $s$ 个 stage 的 circular SMEM buffer 协调生产与消费；异步 load 的发起不会等待其他 load 完成，前 $s$ 次迭代用于填充 buffer。[FA3 §2.2、§3.1、Algorithm 1，PDF pp.3-5](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=3)

NVIDIA Hopper Tuning Guide 还说明，TMA 可处理 1D 到 5D tensor copy，避免用寄存器和普通 SM 指令搬运数据，并允许单个线程发起后让 thread block 继续执行。[NVIDIA Hopper Tuning Guide §1.4.1.2](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)

**工程解释：** TMA 的价值不只是“带宽更高”。它把地址生成、搬运和计算从同一批 warps 的指令流中拆开，使 producer 可以提前填充 $K/V$ stage，consumer 只在真正读取时等待。若没有双缓冲或多缓冲、barrier 和足够的独立计算，异步 copy 仍可能退化成“发起后立刻等待”。

### 3.3 WGMMA：异步的 warpgroup 级 Tensor Core GEMM

WGMMA 是 Hopper 暴露 Tensor Core 的 warpgroup-wide matrix multiply-accumulate 指令族。对 FA3 而言有三个关键点：

1. 由 4 个连续 warps 共同参与；
2. 指令是异步的，发起、commit 与 wait 分离；
3. operand 可以直接来自 SMEM，accumulator 通常位于 registers。

FA3 用两种简写描述 operand 来源：

- **SS-GEMM**：$Q_iK_j^\top$ 的两个输入都来自 shared memory；
- **RS-GEMM**：$\widetilde P_i^{(j)}V_j$ 的第一个输入来自 registers，第二个输入来自 shared memory。

**论文事实：** FA3 Algorithm 1 在 consumer 中用 SS-WGMMA 计算 score，用 RS-WGMMA 累加输出；Algorithm 2 用“commit but do not wait”把 WGMMA 发起与结果消费分开。[FA3 §3.1-§3.2、Algorithm 1-2，PDF pp.5-7](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=5)

**工程解释：** “异步 WGMMA”不表示 score 在发出指令时已经可读。它表示 warpgroup 可以先把 Tensor Core 工作提交到异步执行域，再执行与结果无依赖的普通指令；在读取 accumulator 前仍必须 wait。FA3 的难点正是寻找这些无依赖指令，并把 wait 推迟到最后一个合法位置。

### 3.4 warp specialization 与动态寄存器重分配

一个 FA3 forward CTA 至少包含两类角色：

- producer warpgroup：发起 TMA，维护 circular buffer 的生产端；
- consumer warpgroups：发起 WGMMA，执行 softmax 和输出累加。

producer 不需要保存大型 GEMM accumulator，consumer 却需要大量 registers。Hopper 的 `setmaxnreg` 允许不同 warpgroups 动态交出或获取寄存器配额。

**论文事实：** FA3 §2.2 和 Algorithm 1 明确让 producer deallocate registers、consumer reallocate registers；论文指出 TMA 只需单个线程发起，因此 producer 不应占用与 consumer 相同的寄存器预算。[FA3 §2.2、Algorithm 1](https://arxiv.org/html/2407.08608v2#S3.SS1)

**工程解释：** warp specialization 同时改变了两种资源分配：

1. 指令职责：搬运 warp 不再混杂大量计算指令，计算 warp 不再承担全套 copy 指令；
2. 寄存器职责：少用寄存器的 producer 把预算让给保存 $S$、$\widetilde P$ 和 $\widetilde O$ 的 consumer。

它不是免费优化。专用 producer 占用线程，barrier 占用指令与状态，SMEM 多阶段 buffer 占用容量；只有隐藏掉的延迟超过这些成本时才有净收益。

### 3.5 异步、并发与并行不是同一个词

**工程解释：**

- **并行**：多个独立任务同时由不同执行资源推进，例如不同 CTAs 位于不同 SM；
- **并发**：多个任务都处于可推进状态，硬件交错执行它们；
- **异步发起**：发起方不必等操作完成即可继续，但后续读取结果前仍需同步；
- **overlap**：两类操作在时间轴上确实重叠，通常还要求它们使用不同硬件或存在可并发的执行槽。

因此，只看到 `async`、多个 stage 或多个 warps，不足以证明有 overlap。FA3 Appendix B.2 直接检查 SASS，确认 `MUFU.EX2`、类型转换等指令与部分 HGMMA 指令交错，才构成对实际重叠的证据。[FA3 Appendix B.2，PDF p.19](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=19)

---

## 4. Tensor Core 与 SFU 的吞吐差距

### 4.1 为什么少量 exp 也会很贵

**论文事实：** H100 SXM5 在论文使用的 1830 MHz 时钟下，FP16 Tensor Core 理论吞吐为 989 TFLOP/s；exp 等 special functions 的吞吐约为 3.9 万亿次操作每秒。对 head dimension $d=128$ 的 attention forward，matmul FLOPs 数约为 exp 操作数的 512 倍，但 matmul 峰值也约为 special-function 峰值的 256 倍，因此 exp 时间可以达到 matmul 时间的约 50%。[FA3 §3.1 Pingpong scheduling，PDF pp.5-6](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=5)

**推导：** 令 exp 操作数为 $E$，matmul FLOPs 为 $512E$。只按论文峰值估算：

$$\frac{t_{\exp}}{t_{\mathrm{matmul}}}\approx\frac{E/(3.9\times10^{12})}{512E/(989\times10^{12})}=\frac{989}{512\times3.9}\approx0.50$$

这不表示 softmax 占总时间恰好三分之一，因为 max、sum、scale、类型转换、依赖停顿与实际吞吐也要计入；它只说明“exp 数量少”不能推出“exp 时间可忽略”。

### 4.2 FP8 为什么让这个矛盾更尖锐

**论文事实：** Hopper 的 FP8 WGMMA 吞吐约为 FP16/BF16 的 2 倍，而 exp 吞吐不变。[FA3 §2.2、§3.1](https://arxiv.org/html/2407.08608v2#S2.SS2)

**推导：** 在同一简化模型中把 matmul 峰值翻倍，则：

$$\frac{t_{\exp}}{t_{\mathrm{matmul,FP8}}}\approx\frac{2\times989}{512\times3.9}\approx0.99$$

**工程解释：** 低精度让 Tensor Core 更快，却没有让 softmax 同步变快。如果仍串行执行，Tensor Core 越快，非 GEMM 部分在总时间中的占比反而越高。因此 FA3 的 FP8 路径必须同时处理“更快的 GEMM”和“更难隐藏的 softmax”，否则理论上的 2 倍吞吐不会转化为端到端 2 倍。

### 4.3 “SFU 是瓶颈”应怎样准确理解

**工程解释：** 这里的 SFU 可以理解为执行 `exp2` 等特殊函数的硬件路径，FA3 的 SASS 示例中对应 `MUFU.EX2`。不能把整段 softmax 都归因于一个单元：

- row max 和 row sum 还涉及比较、shuffle 与加法；
- online softmax 还要缩放 $\ell$ 和 $\widetilde O$；
- FP32 -> FP16/FP8 转换也消耗 issue slots 和 registers；
- 真正的停顿取决于依赖链、编译器调度与 active warpgroups。

论文的重点是吞吐不平衡与独立执行资源提供了 overlap 机会，不是声称所有 shape 都严格由 exp 单独限制。

---

## 5. FA2 在 Hopper 上的瓶颈

### 5.1 先区分 FA2 已解决和未解决的问题

FA2 已解决：

- 不把 $S,P\in\mathbb{R}^{N\times N}$ 写入 HBM；
- forward 沿 query tiles 创建更多 CTAs；
- backward 沿 key/value column tiles 创建 CTAs；
- 采用 sliced-Q，减少同一输出的 warp 间 partial-sum 归约；
- 减少 online softmax 中不必要的反复归一化。

**论文事实：** FA2 在 A100 上相对 FA1 约快 2 倍，达到理论峰值的 50%-73%；FA2 论文 Figure 7 还显示，同一实现不使用 TMA、第四代 Tensor Core 等 Hopper 专属能力时，在 H100 上 forward+backward 最高约 335 TFLOP/s。[FA2 Abstract、§4.1、Figure 7，PDF pp.1、10-13](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf#page=10)

FA2 尚未在算法模型中解决：

- copy 与 compute 的专职分工；
- TMA 多阶段预取；
- WGMMA 发起与结果消费的分离；
- softmax 与 Tensor Core GEMM 的显式重叠；
- FP8 的布局和误差控制。

### 5.2 FA3 论文对瓶颈的直接判断

**论文事实：** FA3 报告 FA2 在 H100 上约 35% 利用率，而优化 GEMM 可达到 80%-90%。论文给出两个层次的原因：

1. 实现层面：未用 Hopper 专属指令替代面向 Ampere 的路径；
2. 算法层面：FA2 使用简化的同步模型，没有显式利用异步执行和低精度。

[FA3 Abstract、§1，PDF pp.1-2](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=1)

**工程解释：** FA2 已把主要 HBM 中间量消掉后，瓶颈会向片上调度移动。典型 tile 仍有以下串行依赖：

```text
load K_j -> QK_j^T -> row max / exp / row sum -> P_j V_j -> rescale output
load V_j ---------------------------------------> P_j V_j
```

`load V_j` 不依赖 softmax，可以在 `P_j V_j` 开始前独立预取；真正的计算依赖是 $QK_j^\top \rightarrow \mathrm{softmax} \rightarrow P_jV_j$。若每一步都在前一步完成后才开始，TMA、Tensor Core、MUFU/CUDA cores 中会有资源轮流空闲。FA3 不是删除依赖，而是把不同 tile、不同 query blocks 或不同 warpgroup 的独立工作插进这些空档。

### 5.3 不要把 35% 简化成“FA2 没用 Tensor Core”

**论文事实：** FA2 本来就依赖 Tensor Core GEMM；问题是没有充分使用 Hopper 的异步 WGMMA/TMA 和更细的调度能力。FA3 的对照中还包含“使用 H100-specific instructions 的 Triton FA2”，FA3 仍可更快。[FA3 §4、Figure 5，PDF pp.9-10](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=9)

**工程解释：** “生成了 Tensor Core 指令”和“持续喂满 Tensor Core”是两回事。前者是指令选择，后者还取决于数据是否提前到达、依赖 wait 是否过早、softmax 能否被隐藏、register/SMEM 是否限制 active CTAs，以及 tile 是否适配 WGMMA 布局。

---

## 6. 第一层重叠：producer-consumer 与 inter-warpgroup ping-pong

### 6.1 circular SMEM buffer 怎样工作

设 SMEM 中有 $s$ 个 stage。producer 与 consumer 对第 $j$ 个 $K/V$ tile 执行：

```text
producer:
  等待 stage[j % s] 已被消费
  TMA load K_j, V_j -> stage[j % s]
  到达完成 barrier，通知 consumer

consumer:
  等待 stage[j % s] 已填满
  从该 stage 发起 QK^T 与 PV 的 WGMMA
  确认不再读取后，释放 stage[j % s]
```

**论文事实：** 这是 FA3 Algorithm 1 的 producer-consumer 主循环。$Q_i$ 也由 producer 先搬到 SMEM；producer 在前 $s$ 次迭代不需要等待旧 stage 被释放，因为 buffer 尚未绕回。[FA3 Algorithm 1，PDF p.5](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=5)

**推导：** circular buffer 的 SMEM 开销与 $s$、tile 大小线性相关，大致包含 $s$ 份 $K_j,V_j$ storage。增加 $s$ 可以提供更多预取距离，但会减少同一 SM 可同时容纳的 CTA 数，甚至使 kernel 无法 launch。

**工程解释：** “producer-consumer”主要隐藏 HBM/GMEM 到 SMEM 的搬运与指令发起延迟。它还没有自动解决同一个 consumer 内部 `QK -> softmax -> PV` 的依赖，因此 FA3 还需要下面两种 GEMM-softmax overlap。

### 6.2 inter-warpgroup overlapping 是什么

一个 CTA 中配置多个 consumer warpgroups，让它们处理相互独立的 query 工作。两组 consumer 交替：

```text
时间段 A:
  consumer WG 0: 执行 softmax
  consumer WG 1: 发起并推进 WGMMA

时间段 B:
  consumer WG 0: 发起并推进 WGMMA
  consumer WG 1: 执行 softmax
```

**论文事实：** FA3 使用 `bar.sync` 约束两个 consumer warpgroups 的 GEMM 发起顺序，使 warpgroup 1 的 softmax 尽量落在 warpgroup 2 的 GEMM 时间内，然后交换角色。论文称之为 ping-pong scheduling，并在 Figure 1 展示时间线。[FA3 §3.1、Figure 1，PDF pp.5-6](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=5)；[arXiv HTML §3.1](https://arxiv.org/html/2407.08608v2#S3.SS1)

**论文事实：** 对 FP16 forward、$d=128$、sequence length 8192，论文称 ping-pong 一般可把性能从约 570 TFLOP/s 提升到 620-640 TFLOP/s。论文同时提醒，实际调度没有示意图那么整齐。[FA3 §3.1 Pingpong scheduling，PDF p.6](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=6)

### 6.3 为什么它能跨过同一 tile 的依赖

对一个 query tile，`softmax(S_i^{(j)})` 必须等 $S_i^{(j)}=Q_iK_j^\top$；这是不能删除的真依赖。但另一个 consumer warpgroup 的 score 或 output GEMM 属于另一个独立工作项，不依赖当前 warpgroup 的 softmax。

**推导：** 若 warpgroup A 的 softmax 与 warpgroup B 的 GEMM 没有读写同一 accumulator 或 buffer stage，就可以在不同执行单元上重叠。算法没有让“依赖自己的 score 的 softmax”提前，而是用另一个 warpgroup 的独立 GEMM 填补等待窗口。

**工程解释：** inter-warpgroup overlap 使用的是“任务级独立性”。代价是同时保留多个 consumer 上下文，增加 register 和调度压力。若 head dimension、tile 或 batch 很小，额外 warpgroup 的开销未必值得。

---

## 7. 第二层重叠：intra-warpgroup GEMM/softmax pipelining

### 7.1 原始依赖链为什么看似无法流水

对同一 key tile $j$：

```text
S_j = Q K_j^T
P_j = online_softmax_update(S_j)
O  += P_j V_j
```

$P_j$ 依赖 $S_j$，而第二个 GEMM 又依赖 $P_j$。如果只看一个 tile，三步确实必须按顺序执行。

**工程解释：** 软件流水线的通用做法不是违反单次迭代依赖，而是把不同迭代错开：当 tile $j$ 做 softmax 时，让 Tensor Core 处理 tile $j-1$ 的 $PV$ 或 tile $j+1$ 的 $QK^\top$。

### 7.2 FA3 的两阶段稳态

FA3 Algorithm 2 先做 prologue：等待 $Q_i,K_0$，计算并等待 $S_{\mathrm{cur}}=Q_iK_0^\top$，再计算第一块 softmax。进入稳态后，对尚未进入 epilogue 的 tiles 重复：

1. 异步发起下一块 $S_{\mathrm{next}}=Q_iK_{\mathrm{next}}^\top$，commit 但不立即 wait；
2. 异步发起当前块 $\widetilde O_i\mathrel{+}=\widetilde P_{\mathrm{cur}}V_{\mathrm{cur}}$，commit 但不立即 wait；
3. 在必须读取 $S_{\mathrm{next}}$ 时才等待第一个 WGMMA；
4. 根据 $S_{\mathrm{next}}$ 更新 $m_i,\ell_i,\widetilde P_{\mathrm{next}}$；
5. 在必须缩放输出 accumulator 前才等待第二个 WGMMA；
6. 释放已消费的 $K/V$ buffer stages，把 `next` 状态推进为新的 `cur`。

最后由 epilogue 排空仍在流水线中的 score/probability 与 output GEMM，再完成最终缩放、计算 $L_i$ 并写回。实现时应依据“哪个 score 已生成、哪个 probability 已生成、哪个 $PV$ 尚未消费”这三个状态写出边界，而不是脱离 prologue/epilogue 直接照抄循环上下界。

**论文事实：** Algorithm 2 将前一迭代的第二个 WGMMA 与后一迭代的 softmax 重叠。Figure 2 把它称为 2-stage WGMMA-softmax pipelining。[FA3 §3.2、Algorithm 2、Figure 2，PDF pp.6-7](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=6)；[arXiv HTML §3.2](https://arxiv.org/html/2407.08608v2#S3.SS2)

### 7.3 为什么延后 output rescale 仍然正确

online softmax 在新最大值出现时，需要把历史输出 accumulator 乘以 $\exp(m_{\mathrm{old}}-m_{\mathrm{new}})$。FA3 可以先让异步 WGMMA 把使用旧尺度的 $\widetilde P_{\mathrm{cur}}V_{j-1}$ 累加进去，再在该 WGMMA 完成后把“全部旧尺度贡献”一起 rescale。

**推导：** 设 WGMMA 完成后的旧尺度 accumulator 为 $A_{\mathrm{old}}+C_{\mathrm{cur}}$，新最大值对应缩放因子为 $\gamma=\exp(m_{\mathrm{old}}-m_{\mathrm{new}})$，则延后缩放得到 $\gamma(A_{\mathrm{old}}+C_{\mathrm{cur}})=\gamma A_{\mathrm{old}}+\gamma C_{\mathrm{cur}}$。只要 $A_{\mathrm{old}}$ 与 $C_{\mathrm{cur}}$ 确实使用同一旧基准，延后一次整体缩放不改变实数算术结果。

**工程解释：** 这类变换必须以 accumulator 的“指数基准”作为不变量。若错误地把使用新基准产生的 contribution 也乘以 $\gamma$，或者在 WGMMA 尚未完成时读取/缩放 accumulator，就会得到错误结果或数据竞争。

### 7.4 两阶段流水线的寄存器成本

**论文事实：** 两阶段流水线必须额外保存 $S_{\mathrm{next}}$，每个 CTA 增加 $B_rB_c\operatorname{sizeof}(\mathrm{float})$ 的 register 需求。它可能与大 tile 冲突，并导致 spilling。[FA3 §3.2 Register pressure，PDF p.7](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=7)

**推导：**

- $B_r=64,B_c=128$ 时，一份 FP32 `S_next` 是 32 KiB；
- $B_r=128,B_c=128$ 时，一份 FP32 `S_next` 是 64 KiB。

这些容量分散在 CTA 的 consumer threads 寄存器中，不是一块普通连续 buffer，但总预算仍会挤压 occupancy。

### 7.5 为什么三阶段没有更快

**论文事实：** Appendix B.3 尝试把 tile $j$ 的第二个 WGMMA、tile $j+1$ 的 softmax 和 tile $j+2$ 的第一个 WGMMA 组成三阶段流水。实际表现比两阶段差，原因包括：

- 编译器只把 softmax 与第一个 WGMMA 重叠，没有按预期再与第二个 WGMMA 重叠；
- 还要额外保存一份 $\widetilde P$ 和 `scale_o`；
- 更高 register pressure 迫使实现选更小 tile。

[FA3 Appendix B.3、Algorithm 4、Figure 8，PDF pp.20-21](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=20)

**工程解释：** pipeline depth 不是越大越好。更深流水线提高理论并发度，也同时增加 live ranges、register 数、prologue/epilogue 成本和编译器调度难度。最终应比较实测时间，而不是只比较 stage 数。

### 7.6 两种 overlap 不应混为一谈

| 对比项 | Inter-warpgroup ping-pong | Intra-warpgroup 2-stage pipeline |
|---|---|---|
| 独立性来源 | 不同 consumer warpgroups 的独立 query 工作 | 同一 warpgroup 的不同 key/value tile 迭代 |
| 主要手段 | `bar.sync` 影响 warpgroup 间调度顺序 | WGMMA commit 后延迟 wait，保存 `S_next` |
| 重叠对象 | 一组的 softmax与另一组的 GEMM | 当前/相邻迭代的 softmax 与 GEMM |
| 主要资源成本 | 更多 consumer 上下文和同步 | 额外 score tile registers |
| 论文证据 | §3.1、Figure 1 | §3.2、Algorithm 2、Figure 2、Appendix B.2 |

**论文事实：** Table 2 在固定配置 $\{B,N,H,d\}=\{4,8448,16,128\}$、non-causal FP16 forward 上做消融。完整方案为 3.538 ms、661 TFLOP/s；保留 warp specialization 但去掉 GEMM-softmax pipeline 为 4.021 ms、582 TFLOP/s；保留 pipeline 但去掉 warp specialization 为 4.105 ms、570 TFLOP/s。[FA3 §4.2、Table 2，PDF pp.9、11](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=11)

**推导：** 相对完整方案，两种删减分别慢约 13.7% 和 16.0%；但这不是两个收益可以直接相加，因为两项优化会互相影响调度和资源占用。该表消融的是包含 producer-consumer 与 ping-pong 在内的 warp-specialization 整体，不能从中单独算出 ping-pong 的净收益；§3.1 的 570 -> 620-640 TFLOP/s 才是论文对 ping-pong 的直接量级描述。

---

## 8. FP8：不仅是把输入类型变小

### 8.1 FP8 带来的收益和新误差

**论文事实：** FA3 使用 Hopper 的 FP8 Tensor Core，FP8 WGMMA 的每 SM 吞吐约为 FP16/BF16 的 2 倍。论文讨论的 FP8 格式是 E4M3，即 4 个 exponent bits、3 个 mantissa bits，另有符号位。较短的 mantissa 会增大量化误差，大模型中的 outlier features 又会扩大单一 scale 覆盖的动态范围。[FA3 §2.2、§3.3，PDF pp.4、8-9](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=4)

**工程解释：** FA3 FP8 路径同时面对三类问题：

1. 数值表示：怎样选择 scale，避免少数 outliers 浪费大部分 FP8 code points；
2. WGMMA 布局：两个连续 GEMM 对 operand 的布局要求不一致；
3. 调度：Tensor Core 翻倍后，softmax 更难被隐藏。

### 8.2 FP8 WGMMA 的 k-major 限制

考虑 WGMMA 计算 $A B^\top$，其中 $A\in\mathbb{R}^{M\times K}$、$B\in\mathbb{R}^{N\times K}$：

- k-major：内层归约维 $K$ 连续；
- mn-major：外层 $M$ 或 $N$ 维连续。

**论文事实：** FP16 WGMMA 的 SMEM operands 可接受 mn-major 或 k-major；FP8 WGMMA 只接受 k-major。attention 的 $Q,K,V$ 通常在 head dimension 上连续，但第二个 GEMM $\widetilde P V$ 要求 $V$ tile 在 sequence 维连续，因此不能直接复用原始 $V$ 布局。[FA3 §2.2、§3.3，PDF pp.4、8](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=4)

**工程解释：** 第一个 GEMM 的归约维是 head dimension，原始 feature-last $Q,K$ 很自然；第二个 GEMM 的归约维是 key sequence tile，所需连续方向发生了变化。这是 back-to-back GEMM fusion 中的布局冲突，不是数学上的转置错误。

### 8.3 FA3 怎样处理 $V$ 和 $\widetilde P$ 的布局

对 $V$ 有两个总体选项：

1. kernel 外预转置 $V$；
2. TMA 把普通布局的 $V$ tile 搬进 SMEM 后，在 kernel 内转置。

**论文事实：** FA3 选择 kernel 内转置，使用 warp 协作的 LDSM/STSM 在 SMEM 与 registers 之间搬运并改变布局。除第一轮外，下一块 $V$ 的转置可以隐藏在涉及前一块 $V$ 与当前 $K$ tile 的两次 WGMMA 执行期间。[FA3 §3.3，PDF p.8](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=8)

另一个冲突来自第一个 FP8 WGMMA 的 FP32 accumulator 布局与第二个 FP8 WGMMA 的 register operand A 布局不同。

**论文事实：** FA3 用 byte-permute 指令重排每个线程持有的寄存器元素，并对 $V$ tile 施加匹配的行置换，从而让第二个 WGMMA 得到正确逻辑结果。Figure 3-4 给出两种寄存器 fragment layout。[FA3 §3.3、Figure 3-4，PDF p.8](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=8)

**工程解释：** 两边施加匹配置换利用了矩阵乘的等价性。若 $\Pi$ 是置换矩阵，则：

$$\left(\widetilde P\Pi^\top\right)\left(\Pi V\right)=\widetilde P\left(\Pi^\top\Pi\right)V=\widetilde PV$$

所以物理寄存器顺序可以改变，只要 $\widetilde P$ 的列置换与 $V$ 的行置换互相抵消。论文的具体 byte permutation 是硬件 layout 细节，不能仅凭逻辑 Tensor shape 推出。

### 8.4 Block quantization

常见 per-tensor quantization 为整个 $Q$、$K$ 或 $V$ 各使用一个 scale。FA3 改为：

- 每个 $Q_i\in\mathbb{R}^{B_r\times d}$ block 一个 scale；
- 每个 $K_j,V_j\in\mathbb{R}^{B_c\times d}$ block 各一个 scale。

**论文事实：** FA3 §3.3 指出，块量化可以融合到 attention 前的 memory-bound 操作，例如 rotary embedding；FA3 本来就逐块计算，因此可在 score block 上应用相应 scale，而不增加单独的 attention 计算阶段。[FA3 §3.3 Accuracy，PDF pp.8-9](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=8)

**推导：** 用一个简化对称量化模型表示 $Q_i\approx s_{Q_i}\widehat Q_i$、$K_j\approx s_{K_j}\widehat K_j$，则当前 score tile 为：

$$S_i^{(j)}=\alpha Q_iK_j^\top\approx\alpha s_{Q_i}s_{K_j}\widehat Q_i\widehat K_j^\top$$

每个 tile 独立的 $s_{Q_i}s_{K_j}$ 可以在 score 进入 online softmax 前应用。比起全 Tensor 共用 scale，局部 block 的最大幅值通常更小，绝大多数普通值可使用更多有效 FP8 档位。

**工程解释：** “无额外 slowdown”依赖融合机会和论文配置，不等于量化在抽象算术上零成本。如果上游算子无法融合 scale 统计与转换，独立 quantization kernel 仍会增加 HBM 流量和 launch 开销。

### 8.5 Incoherent processing

outlier 的问题不是向量范数太大，而是能量集中在少数坐标。FA3 在量化前对 $Q,K$ 施加同一个随机正交变换，把集中能量扩散到更多坐标。

按照本文的列向量约定，令 $R\in\mathbb{R}^{d\times d}$ 满足 $R^\top R=I$，并定义 $q_i'=Rq_i$、$k_j'=Rk_j$，则：

$$q_i'^\top k_j'=q_i^\top R^\top Rk_j=q_i^\top k_j$$

因此实数精确算术下 score 不变。对按行堆叠的矩阵，对应 $Q'=QR^\top$、$K'=KR^\top$。FA3 论文写成 $QM,KM$ 且 $MM^\top=I$，与这里令 $M=R^\top$ 等价。

**论文事实：** FA3 使用随机 $\pm1$ 对角矩阵与 Hadamard matrix 的乘积，而不是一般 dense 正交矩阵，使变换复杂度从 $O(d^2)$ 降到 $O(d\log d)$；论文称该变换也可融合进 rotary embedding。[FA3 §3.3 Incoherent processing，PDF p.9](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=9)

**工程解释：** 正交变换不改变 dot product，却会改变坐标分布。一个极大的单坐标经过随机符号与 Hadamard mixing 后，通常分散到许多幅值较温和的坐标，使 FP8 block scale 不再被单个坐标完全支配。这里变换 $Q,K$ 即可保持 score；不能未经补偿地只变换一边。

### 8.6 FP8 不再是“无近似”的同一含义

**论文事实：**

- FA2 和 FA3 的 FP16/BF16 tiled/online-softmax 重排在实数算术下仍以 exact dense attention 为目标；
- FP8 路径明确进行量化，因此相对未量化输入引入近似；
- block quantization 和 incoherent processing 是降低该近似误差，不是消除误差。

**工程解释：** “FlashAttention 是 exact attention”指它不通过稀疏化、低秩化或截断 keys 改变 attention 定义。它从不承诺不同浮点执行顺序逐 bit 相同；FP8 又额外加入了显式量化误差。这两个层面的“exact”必须分开。

---

## 9. Forward 与 backward 的论文支持边界

| 路径 | FA3 论文证据 | 可以得出的结论 | 不能据此声称 |
|---|---|---|---|
| FP16/BF16 forward | §3.1 Algorithm 1、§3.2 Algorithm 2、Figure 5 | 有 producer-consumer、ping-pong、两阶段流水与完整性能评测 | 任意 shape 都有 1.5-2 倍 |
| FP16/BF16 backward | Appendix B.1 Algorithm 3、Figure 6 | 有 warp-specialized backward 和非 causal 性能数据 | 论文给出了与 forward 相同的两阶段流水细节 |
| FP8 forward | §3.3、Figure 7/9、Table 3 | 有布局处理、块量化、不相干处理、性能与误差评测 | FP8 仍与 FP16 一样“无量化近似” |
| FP8 backward | 无对应算法与 benchmark | 论文没有建立支持证据 | 根据 FP8 forward 自动推断 FP8 backward |
| Causal forward | Figure 5、Figure 7/9 | FP16 与 FP8 都有 causal 性能曲线 | causal 下总能达到 non-causal 的同等 TFLOP/s |
| Causal backward | Figure 6 只展示 non-causal | 论文未给 causal backward 曲线 | 从 forward 曲线推断 backward 数字 |
| MQA/GQA | §3.1 Attention variants | 通过索引避免在 HBM 复制 $K,V$ | 论文覆盖所有 KV-cache/decode 优化 |
| LLM decode | §5 列为未来优化方向 | 论文主结果不是 decode 专用 kernel | 把训练/prefill 吞吐直接外推到单 token decode |

**论文事实：** FA3 backward 在 producer 和 consumers 之外增加一个 `dQ-writer` warp。不同 CTAs 对同一 $dQ_i$ 有贡献，writer 从 SMEM 取本地 $dQ_i^{(\mathrm{local})}$，再借助 semaphore 原子累加到 global memory，使 consumers 可以继续下一轮 matmul。[FA3 Appendix B.1、Algorithm 3，PDF p.18](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=18)

**工程解释：** forward 每个 query tile 独占 $O_i$，天然无跨 CTA 写冲突；backward 以 key/value tile 为所有权单位时，每个 CTA 独占 $dK_j,dV_j$，但多个 CTAs 会共同贡献 $dQ_i$。专用 writer 的意义是隐藏这条有争用的写回路径，而不是改变梯度公式。

**论文事实：** FA3 §5 还把三项内容列为局限或未来工作：优化 LLM inference、为 FP8 kernel 集成 persistent design、理解低精度 attention 对大规模训练的影响。论文脚注说明 benchmark 中 FP16 FA3 有 persistent kernel 与 load balancing，而 FP8 FA3 没有，这部分解释了 FP8 在短序列和 causal 场景不如 cuDNN 的现象。[FA3 §5，PDF p.12](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=12)

---

## 10. 性能数字应该怎样读

### 10.1 Benchmark 口径

**论文事实：** FA3 Appendix C.1 的主要环境为：

- H100 80GB SXM5，功耗上限 700W；
- GPU 时钟固定为 1830 MHz；
- CUDA 12.3、cuDNN 9.1.1.17、CUTLASS 3.5；
- FlashAttention 2.5.8；
- Triton nightly `3.0.0.post20240424212437`；
- PyTorch 2.3.0；
- 重复 100 次并取平均。

[FA3 Appendix C.1，PDF p.21](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=21)

主 benchmark 固定总 token 数为 16k，sequence length 从 512 变化到约 16k，hidden dimension 为 2048，head dimension 为 64、128 或 256。按完整 batch 的实际统计口径：

$$\mathrm{FLOPs}_{\mathrm{fwd}}=4BHN^2d$$

非方形注意力（query 长度为 $N_q$、key 长度为 $N_k$）对应：

$$\mathrm{FLOPs}_{\mathrm{fwd}}=4BHN_qN_kd$$

causal 时除以 2；backward 按 forward 的 2.5 倍计算，因为 forward 有 2 个 matmuls，带重算的 backward 有 5 个 matmuls。[FA3 §4.1](https://arxiv.org/html/2407.08608v2#S4.SS1)

**勘误说明：** FA3 §4.1 的公式排版漏写了 batch size $B$。Table 2 中 $B=4$、$N=8448$、$H=16$、$d=128$、$t=3.538\ \mathrm{ms}$，使用包含 $B$ 的完整公式计算，forward 吞吐约为 $661\ \mathrm{TFLOP/s}$。

**工程解释：** 图中的 TFLOP/s 是用约定 FLOPs 除以时间得到的有效吞吐，不是硬件计数器直接测得的指令 FLOPs。不同论文、不同 causal FLOPs 口径、时钟、版本或 shape 的数字不能直接横比。

### 10.2 论文 headline 与代表性数据

| 项目 | 论文报告 |
|---|---:|
| FA3 FP16 forward 相对 FA2 | 1.5-2.0 倍 |
| FA3 FP16 backward 相对 FA2 | 1.5-1.75 倍 |
| FA3 FP16 headline | 最高约 740 TFLOP/s，约为 989 TFLOP/s 理论峰值的 75% |
| FA3 FP8 headline | 接近 1.2 PFLOP/s |
| FA2 在 H100 的利用率描述 | 约 35% |

来源：[FA3 Abstract、§1、§4.1](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=1)。

从 Figure 5-7 读取几个固定点，可建立量级直觉：

| dtype / pass / shape | FA2 | FA3 | 说明 |
|---|---:|---:|---|
| FP16 forward，non-causal，$d=128,N\approx8k$ | 370 | 646 | TFLOP/s |
| FP16 forward，causal，$d=128,N\approx8k$ | 333 | 602 | TFLOP/s |
| FP16 backward，non-causal，$d=128,N\approx8k$ | 318 | 559 | TFLOP/s |
| FP8 forward，non-causal，$d=256,N\approx8k$ | 未列 FA2 FP8 | 1151 | TFLOP/s |
| FP8 forward，non-causal，$d=256,N\approx16k$ | 未列 FA2 FP8 | 1171 | TFLOP/s |

来源：[FA3 Figure 5，PDF p.10](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=10)、[Figure 6-7，PDF p.11](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=11)。

**工程解释：** 图上个别点达到约 756 TFLOP/s，而摘要用“up to 740 TFLOP/s”概括。写报告时应注明是引用摘要 headline 还是读取具体曲线，不要把四舍五入差异误判成冲突。

### 10.3 性能不是只由算法名字决定

同一 Figure 5 同时比较了：

- 普通 attention；
- FA2 CUDA；
- 使用 H100 专属指令的 Triton 实现；
- cuDNN；
- FA3。

**论文事实：** 对中长序列，FA3 FP16 可超过论文中的 cuDNN 版本；FP8 与 cuDNN 的相对关系依赖 head dimension 和 causal mask，论文脚注明确指出 $d=64$ 时 FA3 领先，而 $d=128,256$ 时 non-causal 大致相当、causal 落后。[FA3 §1 脚注 2、§4.1、Figure 5、Figure 9](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=2)

**工程解释：** 影响结果的变量至少包括 dtype、$N$、$d$、causal、tile、active CTAs、persistent scheduling、软件版本和时钟。合理结论是“FA3 的调度在论文配置上显著提升 H100 利用率”，而不是“任何名为 FA3 的 kernel 永远比任意 vendor kernel 快”。

---

## 11. 数值误差：表 3 真正证明了什么

### 11.1 实验设置

**论文事实：** FA3 §4.3 以 FP64 attention 为 reference。为模拟 outliers，$Q,K,V$ 的普通元素来自标准正态分布，另有 0.1% 的位置叠加标准差为 10 的正态项。指标是输出的 RMSE。[FA3 §4.3、Table 3，PDF pp.10-12](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=10)

论文中的 FP8 baseline 使用 per-tensor scaling、FP32 matmul accumulator，并把 softmax 中间结果保留为 FP16；因此下面比较的是两套完整数值路径，不是只改变一个输入 dtype。

### 11.2 FP16 结果

| 方法 | RMSE |
|---|---:|
| Baseline FP16 attention | $3.2\times10^{-4}$ |
| FA2 FP16 | $1.9\times10^{-4}$ |
| FA3 FP16 | $1.9\times10^{-4}$ |

**论文事实：** 论文将 FA2/FA3 约 1.7 倍更低的 RMSE 归因于 softmax rescaling 等中间结果保留在 FP32。[FA3 Table 3](https://arxiv.org/html/2407.08608v2#S4.SS3)

**工程解释：** 浮点操作次序改变不必然降低精度，但“关键归约与状态使用 FP32”可能比某个把中间 softmax 留在 FP16 的 baseline 更准确。因此不能把“fused”自动等同于“精度更差”。

### 11.3 FP8 结果

| 方法 | RMSE |
|---|---:|
| Baseline FP8，per-tensor scale | $2.4\times10^{-2}$ |
| FA3 FP8，完整方案 | $9.1\times10^{-3}$ |
| FA3 FP8，去掉 block quantization | $9.3\times10^{-3}$ |
| FA3 FP8，去掉 incoherent processing | $2.4\times10^{-2}$ |

**推导：** $2.4\times10^{-2}/9.1\times10^{-3}\approx2.64$，对应论文“2.6 倍更低误差”的表述。

**工程解释：** 在这一个合成 outlier 分布中，去掉 incoherent processing 的退化远大于去掉 block quantization。但不能据此断言 block quantization 对所有模型都无用。Table 3 只测一个输入分布、一个误差指标和 attention 输出，没有覆盖训练收敛、梯度误差或真实模型所有层。

### 11.4 应怎样验证自己的 FP8 kernel

至少分四层验证：

1. 用 FP64 reference 测 attention 输出；
2. 分别关闭 block quantization 和 incoherent processing 做消融；
3. 覆盖普通高斯、显式 outlier、真实 activation 三种分布；
4. 分开报告 absolute error、relative error、RMSE、最大误差和下游 loss。

**工程解释：** 只与 FP16 FA2 比较吞吐而不报告量化误差，无法说明 FP8 路径是否可用；只测无 outlier 的均匀随机输入，又可能掩盖 per-tensor scale 的主要失败模式。

---

## 12. 复杂度、I/O 与可并行性

### 12.1 渐近复杂度没有改变

令 batch 为 $B$、heads 为 $H$、query/key 长度为 $N_q,N_k$、head dimension 为 $d$。

| 成本 | FA2 | FA3 |
|---|---|---|
| Forward 主算术 | $O(BHN_qN_kd)$ | 相同 |
| Backward 主算术 | $O(BHN_qN_kd)$，常数更大且重算 $S,P$ | 相同 |
| HBM 中完整 $S,P$ | 不物化 | 不物化 |
| Forward 额外全局状态 | $L$，$O(BHN_q)$ | 相同 |
| Backward 额外全局状态 | $D$ 等线性状态 | 相同量级 |
| CTA 片上状态 | 当前 tiles 与 accumulator | 增加 circular SMEM stages、额外 score buffer 等常数项 |

**论文事实：** FA2 Algorithm 1 明确给出 $O(N^2d)$ FLOPs 和除输入输出外 $O(N)$ memory；FA3 复用相同 tile 循环与 online softmax。[FA2 §3.1.1 Correctness, runtime, and memory requirement](https://arxiv.org/html/2307.08691v1#S3.SS1.SSS1.Px2)；[FA3 §2.3、§3](https://arxiv.org/html/2407.08608v2#S3)

**推导：** FA3 的 $s$-stage SMEM buffer 与额外 `S_next` 都由固定硬件 tile 大小决定，不随完整 $N$ 二次增长，所以不改变关于序列长度的渐近空间复杂度。

### 12.2 FA3 改变的是关键路径与资源重叠

若把每个 tile 的时间粗略拆成搬运 $T_{\mathrm{mem}}$、GEMM $T_{\mathrm{gemm}}$、softmax/非 GEMM $T_{\mathrm{softmax}}$：

- 串行模型接近 $T_{\mathrm{mem}}+T_{\mathrm{gemm}}+T_{\mathrm{softmax}}$；
- 理想充分重叠后接近 $\max(T_{\mathrm{mem}},T_{\mathrm{gemm}},T_{\mathrm{softmax}})$；
- 真实 FA3 还要加 barrier、pipeline fill/drain、layout transform、资源不足和未隐藏尾部。

**推导：** 上式只是关键路径模型，不是论文给出的闭式性能公式。

**工程解释：** FA3 的优化目标是缩短 wall-clock critical path，而不是减少主要 GEMM FLOPs。若只统计 FLOPs，会看不到这类收益；若只统计 occupancy，也看不到 Tensor Core 与 MUFU 是否真正同时工作。

### 12.3 四个并行层次

1. **跨 CTA**：沿 $B,H,\lceil N_q/B_r\rceil$ 分配 forward query tiles，继承 FA2。
2. **CTA 内角色并行**：producer TMA 搬运，consumers 计算。
3. **跨 consumer warpgroups**：ping-pong 重叠一组 softmax 与另一组 GEMM。
4. **单 consumer warpgroup 内**：跨 key/value tile 迭代做两阶段 WGMMA-softmax pipeline。

**工程解释：** 这四层解决不同空闲来源。增加 CTA 数解决 SM 闲置；warp specialization 解决 copy 与 compute 指令混杂；ping-pong 提供独立 consumer 工作；两阶段 pipeline 缩短单个 consumer 的依赖关键路径。

### 12.4 Causal 与短序列为何更难

**论文事实：** FA2 指出 causal mask 可跳过约一半完全位于对角线上方的 blocks，但实际加速约为 1.7-1.8 倍而非严格 2 倍。[FA2 §3.1.1 Causal masking，PDF pp.6-7](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf#page=6)

**工程解释：**

- 对角 tile 仍要做逐元素 mask；
- pipeline 的 prologue/epilogue 不会随有效 tile 数同比缩小；
- causal 导致不同 query tiles 的工作量不均；
- 短序列的 tile 数少，更难摊薄 persistent scheduling、转置和同步开销；
- FP8 FA3 尚无 persistent/load-balancing 设计，论文也据此解释短序列与 causal 场景的劣势。

---

## 13. 对 CUDA/CUTLASS 实现的启示

### 13.1 先画依赖图，再写异步代码

一个实现至少要显式标出：

```text
Q stage ready
K_j stage ready -> QK_j^T issued -> score ready -> softmax state ready
V_j stage ready ---------------------------------> PV issued -> output ready
stage no longer referenced -> producer may overwrite
```

**工程解释：** barrier 应绑定“数据所有权状态”，而不是凭感觉插在循环中。每个 wait 都应回答：

- 等的是 TMA copy、WGMMA group，还是另一个 warpgroup；
- 哪个 register/SMEM 数据即将被读取；
- 哪个 stage 在何时可以安全复用。

过早 wait 会丢失 overlap；过晚 wait 会读取未完成结果；过早释放 stage 会产生覆盖竞争。

### 13.2 把 operand layout 当成算法接口

FA3 FP8 说明，逻辑 shape 相同不代表 WGMMA 可直接消费。实现需要同时记录：

- logical shape；
- SMEM major order 与 swizzle；
- 每个 thread 持有的 register fragment；
- accumulator layout；
- 从第一 GEMM 输出到第二 GEMM operand 的置换；
- $V$ 侧必须匹配的反置换。

**工程解释：** 对 Tensor Core kernel，layout 不是末端代码生成细节，而是跨 pipeline stage 的类型信息。一个更可靠的抽象应让非法 layout 组合尽早失败，而不是等数值测试发现列错位。

### 13.3 用资源模型选择 tile 和 stage

需要共同考虑：

- $B_r,B_c,d$ 决定 WGMMA 形状和 score/accumulator registers；
- SMEM stage 数 $s$ 决定预取距离与 SMEM 占用；
- consumer warpgroup 数决定 ping-pong 机会与 registers；
- 两阶段流水线额外保留 `S_next`；
- 更大 tile 减少循环和通信，却可能 spill 或降低 active CTA 数。

**论文事实：** FA2 已指出大 tile 会增加 register/SMEM，可能 spill 或无法 launch；FA3 再增加 pipeline buffer，使该权衡更紧。[FA2 §3.3 Tuning block sizes](https://arxiv.org/html/2307.08691v1#S3.SS3.SSS0.Px3)；[FA3 §3.2 Register pressure](https://arxiv.org/html/2407.08608v2#S3.SS2.SSS0.Px2)

### 13.4 必须检查生成代码

**论文事实：** FA3 Appendix B.2 通过 SASS 确认：

- softmax 被重排到第一个 WGMMA 前部；
- `MUFU.EX2`、FP32 -> FP16 conversion、row sum 和 output rescale 与第一个 WGMMA 的 HGMMA 指令交错；
- 第二个 WGMMA 没有与其他指令重叠；
- 循环末尾才执行 WGMMA wait。

[FA3 Appendix B.2，PDF p.19](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=19)

**工程解释：** 源码表达了依赖允许的调度空间，但编译器决定实际 instruction schedule。性能验证应包含 PTX/SASS 检查与 profiler timeline，不能只看源码中 `async` 或 `num_stages` 的字样。

---

## 14. 对 Triton 实现的启示

### 14.1 Triton program 与 FA3 CTA 的对应

在 NVIDIA 后端上，可以把一个 Triton program instance 近似理解为一个 CTA 级 tile 任务：

- grid 仍应让不同 programs 拥有不同 query output tiles；
- program 内循环 key/value tiles；
- program 内需要表达数据预取、dot、softmax 和 accumulator；
- 编译器再决定 warps、Tensor Core 指令和软件流水。

**工程解释：** FA3 的论文实现使用 CUTLASS primitives，不意味着把 Algorithm 2 逐行翻成 Triton 就能得到同样调度。Triton 能否生成 TMA/WGMMA、warp-specialized pipeline 与预期 layout，取决于版本、target、descriptor、编译器 pass 和 kernel 形状。

### 14.2 三个常见的错误等价

1. `num_warps=8` 不等于“已有两个 consumer warpgroups 做 ping-pong”；
2. `num_stages=2` 不等于 FA3 的“两阶段 GEMM-softmax pipeline”；
3. `tl.dot` 降成 WGMMA 不等于 softmax 已被隐藏。

**工程解释：**

- `num_warps` 只声明 program 使用的 warp 数，不能单独表达角色和调度；
- memory pipeline stages 管的是 tile copy buffer，FA3 2-stage 管的是 register 中跨迭代的 score/GEMM-softmax 依赖；
- WGMMA 的存在只证明用了对应矩阵指令，不证明 wait 位置、MUFU overlap 或 register layout 最优。

### 14.3 一个更可靠的 Triton 开发顺序

1. 先实现与 FA2 一致的 FP16/BF16 数学，验证 $O,L$ 和 backward；
2. 确认 grid 与 output ownership，避免跨 programs 的非原子覆盖；
3. 使用目标版本支持的 tensor descriptor/async copy 机制表达 HBM -> SMEM；
4. 检查生成 PTX/SASS 是否出现目标 TMA/WGMMA；
5. 用 timeline 验证 copy、WGMMA 与 softmax 是否重叠；
6. 再增加 warp specialization 或更深 pipeline；
7. 最后处理 FP8 scale、$V$ transpose 和 register fragment layout；
8. 每加一项优化都保留消融 benchmark。

Triton 官方 fused-attention tutorial 可用于核对当前版本的 descriptor、warp specialization 和 FP8 分支，但其 API 与调度会随版本变化，不能替代论文中的算法不变量。[Triton Fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### 14.4 推荐 profiler 问题清单

不要只问“TFLOP/s 是否提高”，还应检查：

- Tensor Core active 周期是否提高；
- TMA copy 是否与 consumer compute 重叠；
- WGMMA wait 前是否有足够独立指令；
- MUFU/SFU 与 WGMMA 是否在同一时间窗口活跃；
- register spill 是否出现；
- active CTAs/warpgroups 是否因 SMEM 或 registers 降低；
- barrier stall 是否增加；
- causal 不均衡和 wave quantization 是否造成尾部空闲；
- FP8 transpose/permute 是否被隐藏。

**工程解释：** 若吞吐下降，按上述资源链定位比盲目调 tile 更有效。例如，register spill 应先减 live ranges 或 pipeline depth；TMA wait 长则考虑预取距离；MUFU 暴露则考虑 overlap；尾波明显则考虑 persistent scheduling 或 load balancing。

---

## 15. FA2 与 FA3 完整对比表

| 问题 | FA2 的回答 | FA3 的回答 |
|---|---|---|
| 是否改变 dense attention 定义 | 否 | FP16/BF16 否；FP8 有量化近似 |
| 是否物化完整 $S,P$ | 否 | 否 |
| Forward 输出所有权 | 一个 CTA 负责一个 query tile | 继承 |
| Backward 主要所有权 | 一个 CTA 负责 key/value column tile，$dQ$ 跨 CTA 累加 | 继承，并加入专用 `dQ` writer |
| CTA 内 warp 划分 | sliced-Q，避免 sliced-K partial-output 归约 | producer/consumer warp specialization |
| HBM -> SMEM | tile load，论文算法层未强调 Hopper 异步硬件 | TMA producer + circular SMEM buffer |
| GEMM | Tensor Core matmul | 异步 WGMMA，SS/RS operands |
| softmax 与 GEMM | 主要按 tile 依赖链推进 | inter-warpgroup ping-pong + intra-warpgroup pipeline |
| 寄存器管理 | tile/accumulator 常规权衡 | `setmaxnreg` 在 producer/consumer 间重分配 |
| 低精度 | FP16/BF16 为主 | 增加 FP8 forward |
| FP8 layout | 未处理 | $V$ kernel 内转置、accumulator/operand byte permute |
| FP8 误差控制 | 未处理 | block quantization + incoherent processing |
| 复杂度 | $O(N^2d)$，额外全局内存线性 | 相同渐近量级，片上常数更大 |
| 主要性能证据 | A100 50%-73% 峰值；H100 旧实现未用专属新指令 | H100 FP16 约 75% 峰值，FP8 接近 1.2 PFLOP/s |
| 关键限制 | 没有显式利用 Hopper 异步与 FP8 | 调度和布局更复杂，register/SMEM 压力更高 |

---

## 16. 常见误解

### 16.1 “FA3 使用了新的 attention 公式”

错误。FP16/BF16 主线仍是 FA2 的 tiled online softmax。变化主要在调度、异步硬件和低精度表示。

### 16.2 “异步 WGMMA 会自动与 softmax 重叠”

错误。必须有无数据依赖的工作、延迟 wait，并且编译器实际生成交错指令。FA3 用跨迭代 pipeline 和 SASS 检查证明这一点。

### 16.3 “TMA 就是更快的 `load`”

不完整。TMA 的关键价值包括专用 copy engine、减少普通 SM 指令与 register 搬运、单线程发起、多维描述和与 barrier/pipeline 协作。

### 16.4 “warpgroup 等于 128 线程的普通 warp”

错误。warp 仍是 32 threads；warpgroup 是 4 个连续 warps，WGMMA 以这个协作组为粒度。

### 16.5 “producer warpgroup 的每个线程都在搬数据”

错误。论文指出 TMA 只需单线程发起。保留 producer warpgroup 是角色划分与调度设计，不表示 128 个线程逐元素 copy。

### 16.6 “FA3 的两个 stage 就是双缓冲”

错误。论文同时存在两个不同的 stage 概念：

- $s$-stage circular SMEM buffer：数据搬运流水；
- 2-stage WGMMA-softmax pipeline：register 中跨迭代的计算流水。

两者的 stage 数有上界关系，但不要求相等。[FA3 §3.2 脚注 6](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf#page=6)

### 16.7 “pipeline 越深越快”

错误。论文的三阶段版本反而更慢，原因包括 register pressure、tile 变小和编译器没有产生预期重叠。

### 16.8 “softmax FLOPs 很少，所以无需优化”

错误。不同类型操作的吞吐相差巨大。论文对 $d=128$ 的峰值估算表明，exp 时间可达到 matmul 时间的约 50%。

### 16.9 “FP8 只需要 cast”

错误。还要处理 scale 粒度、outliers、WGMMA k-major 限制、$V$ transpose、FP32 accumulator 到 FP8 operand 的 layout 变换，以及误差验证。

### 16.10 “正交变换会改变 attention score”

错误。对列向量同时使用同一正交变换 $R$ 时，$(Rq)^\top(Rk)=q^\top k$。改变的是坐标分布，不是实数 dot product。

### 16.11 “FA3 FP8 仍是完全无近似”

错误。它不做稀疏或低秩 attention 近似，但 FP8 quantization 本身引入数值近似。

### 16.12 “论文展示 FP8 forward，所以 FP8 backward 也已被证明”

错误。FA3 论文没有给 FP8 backward 算法或 benchmark。

### 16.13 “FA3 对所有 GPU 都会同样加速”

错误。论文认为方法可推广到具有足够异步和低精度能力的加速器，但实证围绕 H100。缺少 TMA/WGMMA 等能力时，收益和实现方式都会变化。

### 16.14 “TFLOP/s 可以跨论文直接比较”

错误。必须对齐 FLOPs 定义、causal 折算、shape、dtype、时钟、软件版本和 forward/backward 口径。

---

## 17. 推荐阅读路线

### 路线 A：只想建立 FA2 -> FA3 主线

1. 本文第 1 节：总览差异；
2. 第 5 节：FA2 在 Hopper 上的瓶颈；
3. 第 6-7 节：两层 overlap；
4. 第 15 节：对比表。

对应论文：

1. [FA2 §3](https://arxiv.org/html/2307.08691v1#S3)；
2. [FA3 §1](https://arxiv.org/html/2407.08608v2#S1)；
3. [FA3 §3.1](https://arxiv.org/html/2407.08608v2#S3.SS1)；
4. [FA3 §3.2](https://arxiv.org/html/2407.08608v2#S3.SS2)。

### 路线 B：准备实现 FP16/BF16 kernel

1. 复习 [03_02 第 4-8 章](./03_02_flash_attention_2_beginner_textbook.md)；
2. 阅读本文第 2-7 节；
3. 逐行对照 FA3 Algorithm 1 与 Algorithm 2；
4. 阅读 Appendix B.2 的 SASS；
5. 按本文第 13-14 节建立 correctness、resource 与 profiler 检查。

### 路线 C：准备实现 FP8 forward

1. 先完成路线 B；
2. 阅读本文第 8 节；
3. 对照 FA3 Figure 3-4 理解 accumulator/operand fragment；
4. 对照 §3.3 实现 $V$ tile transpose 与匹配置换；
5. 分别实现 per-tensor、per-block、incoherent 三条精度 baseline；
6. 用 Table 3 的消融方式验证，而不是只测吞吐。

### 路线 D：研究 backward

1. 复习 [03_03 第 9 章](./03_03_flash_attention_2_backward.md) 的 $L,D$ 与 tile 重算；
2. 对照 FA2 Algorithm 2 理解 key/value tile ownership；
3. 阅读 FA3 Appendix B.1 Algorithm 3；
4. 单独分析 producer、consumers 与 `dQ-writer` 的同步；
5. 不把 forward 的两阶段流水细节未经证明地套到 backward。

---

## 18. 一页复习

| 问题 | 最短答案 |
|---|---|
| FA3 改了 attention 数学吗？ | FP16/BF16 没改；FP8 增加量化近似 |
| FA2 为什么在 H100 上不够快？ | 算法调度模型偏同步，没有显式利用 TMA、异步 WGMMA 和 softmax overlap |
| warpgroup 是什么？ | 4 个连续 warps，共 128 threads |
| TMA 做什么？ | 异步搬运 GMEM/SMEM tensor tiles，让 producer 与 compute 解耦 |
| WGMMA 做什么？ | 由 warpgroup 异步发起 Tensor Core GEMM，可直接读取 SMEM operand |
| producer-consumer 隐藏什么？ | 主要隐藏数据搬运和指令发起延迟 |
| ping-pong 隐藏什么？ | 一组 consumer 的 softmax藏在另一组的 GEMM 下 |
| 两阶段 pipeline 隐藏什么？ | 同一 consumer 中相邻 tile 的 GEMM 与 softmax |
| 两种 stage 是否相同？ | 否，SMEM copy stages 与 GEMM-softmax stages 是不同概念 |
| softmax 为什么值得隐藏？ | Tensor Core 与 exp 特殊函数吞吐差异巨大 |
| FP8 为什么需要转置和置换？ | FP8 WGMMA 只接受 k-major，且两次 GEMM 的 fragment layout 冲突 |
| block quantization 做什么？ | 缩小一个 scale 覆盖的动态范围 |
| incoherent processing 做什么？ | 用正交变换扩散 outlier，同时保持 $q^\top k$ |
| FA3 改变复杂度吗？ | 不改变 $O(N^2d)$ 主算术与线性额外全局内存 |
| 论文有 FP8 backward 吗？ | 没有，只给出并评测 FP8 forward |
| 如何确认 overlap？ | 检查生成 PTX/SASS 和 profiler timeline，而非只看源码 |

---

## 19. 参考资料

1. Tri Dao. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](./references/flash_attention/flashattention_2_better_parallelism_and_work_partitioning.pdf). 2023. [arXiv abstract](https://arxiv.org/abs/2307.08691)，[arXiv HTML](https://arxiv.org/html/2307.08691v1)。
2. Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](./references/flash_attention/flashattention_3_asynchrony_and_low_precision.pdf). 2024. [arXiv abstract](https://arxiv.org/abs/2407.08608)，[arXiv HTML](https://arxiv.org/html/2407.08608v2)。
3. NVIDIA. [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)，重点见 Tensor Memory Accelerator 与 occupancy。
4. NVIDIA. [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，重点见 WGMMA 与 `setmaxnreg`。
5. Triton. [Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
