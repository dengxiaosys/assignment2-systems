# Nsight Systems 单 Profile 实验报告

## 0. 范围与限制

本文对应 handout 的 `nsys_profile` 问题，仅分析一个已经采集并复制回开发机的 profile：

- 报告：[small_b2_s512.nsys-rep](../profile_artifacts/small_b2_s512.nsys-rep)
- 报告导出的 SQLite：[small_b2_s512_profile.sqlite](../profile_artifacts/small_b2_s512_profile.sqlite)
- Profile 同次 Python 结果：[small_b2_s512_profile_result.json](../profile_artifacts/small_b2_s512_profile_result.json)
- 未启用 profiler 的基线结果：[small_b2_s512_benchmark_result.json](../profile_artifacts/small_b2_s512_benchmark_result.json)
- SHA-256：`b5d679fb97c7b6fd24e2193e42e9a5616d52b4f26f82f7712299464d90b388d4`
- GPU：NVIDIA GeForce GTX 1060 6 GiB
- PyTorch：2.11.0+cu126
- 模型：`small`，128,625,408 参数
- 模式：`full`
- dtype：FP32
- batch size：2
- context length：512
- warmup：5 steps
- measurement：1 step

用户明确要求不运行其它模型尺寸，因此本文不满足原题“两个模型尺寸、三个 context length”的 sweep 要求。以下结论只对该配置和硬件成立。

采集脚本见 [run_small_profile.sh](../remote_profile/run_small_profile.sh)，计时实现见 [_run_step](../cs336_systems/benchmark.py#L123-L179)，注意力实现见 [scaled_dot_product_attention](../../assignment1-basics/cs336_basics/model.py#L121-L133)。

---

## 1. 分析方法

### 1.1 从同一个报告导出数据

使用 `nsys stats` 将 `.nsys-rep` 导出为 SQLite：

```bash
nsys stats --report cuda_gpu_kern_sum \
  ~/var/cs336-profile/profiles/small_b2_s512/profile.nsys-rep
```

后续统计只查询由该 profile 导出的 `profile.sqlite`，没有重新运行其它模型。

这四个证据文件均保存在本地 `profile_artifacts/`，该目录已加入 `.gitignore`。SQLite 对应的 `.nsys-rep` 与本地报告具有相同 SHA-256，并非来自另一次 profile。

### 1.2 如何划分阶段

报告中 `benchmark_measurement` NVTX range 的墙钟区间为 496.960 ms。该区间内有 6 次 `cudaDeviceSynchronize()`：

1. measurement step 开始前同步；
2. forward 结束；
3. loss 结束；
4. backward 结束；
5. optimizer 结束；
6. total 计时结束前的最终同步。

Warmup 并非没有同步。[`run_benchmark`](../cs336_systems/benchmark.py#L242-L267) 先执行 warmup，随后才进入 `benchmark_measurement` NVTX range，因此 warmup 的同步事件存在于完整 profile 中，但被本节的 range 过滤条件排除。另一个区别是 warmup 调用 `_run_step(..., measure=False)`，只在整个 step 末尾执行一次同步；正式 measurement 调用 `measure=True`，会在 step 开始、各阶段结束和 total 结束处同步，所以单个 full measurement step 中能看到 6 次同步。

因此可以用相邻同步调用的结束时间划分各阶段。得到的时间与 Python `timeit.default_timer()` 结果一致：

| 阶段 | Nsight 同步边界 (ms) | Python 计时 (ms) |
|---|---:|---:|
| forward | 136.273 | 136.278 |
| loss | 1.761 | 1.749 |
| backward | 289.738 | 289.746 |
| optimizer | 68.056 | 68.042 |
| total | 495.852 | 495.851 |

表中的 136.273 ms 不是 JSON 中的直接字段，而是 SQLite 中“measurement 开始前同步”和“forward 结束同步”两个事件的结束时间之差：`(5156656272 - 5020383439) / 1e6 = 136.272833 ms`。136.278 ms 则直接记录在 `small_b2_s512_profile_result.json` 的 `phases.forward.mean_ms`。

微小差异来自 Python 计时语句与 CUDA API 事件边界之间的主机端开销。

### 1.3 各阶段 GPU kernel 汇总

| 阶段 | GPU kernel 总时间 (ms) | GEMM 时间 (ms) | GEMM 占比 |
|---|---:|---:|---:|
| forward | 131.743 | 84.464 | 64.11% |
| loss | 1.656 | 0 | 0% |
| backward | 281.951 | 171.787 | 60.93% |
| optimizer | 67.404 | 0 | 0% |
| full step | 482.754 | 256.251 | 53.08% |

这里按 kernel 短名称中是否包含 `gemm` 分类。该模型在默认 CUDA stream 上顺序执行，因此 kernel 时长求和可以用于该实验的阶段占比分析。

---

## 2. Handout 问题与回答

### 2.1 问题 (a)

> What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?
>
> **Deliverable:** A 1-2 sentence response.

**回答**

Nsight 同步边界给出的训练 forward 时间为 136.273 ms，同一次运行中的 Python 计时为 136.278 ms；当前保留的未启用 profiler 基线 JSON 中，10 次均值为 136.570 ± 1.020 ms。Nsight 与基线均值相差约 0.22%，可以认为一致。

### 2.2 问题 (b)

> What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?
>
> **Deliverable:** A 1-2 sentence response.

**回答**

Forward 中累计 GPU 时间最高的是 `sgemm_128x128x8_NT_vec`，调用 85 次、累计 75.263 ms；结合模型结构，85 次正好对应每层 7 个线性层 GEMM 乘以 12 层，再加最终 LM head，这一归因并非仅从 kernel 名称得出。Backward 的线性层 GEMM 总数是 170 次，但 Nsight 按 kernel 名称将其拆成 `sgemm_128x128x8_NN_vec` 85 次和 `sgemm_128x128x8_TN_vec` 85 次；前者累计 80.300 ms、单次平均耗时更高，因此它成为 backward 中最高耗时的 kernel，而其分组调用次数仍显示为 85。

### 2.3 问题 (c)

> Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
>
> **Deliverable:** A 1-2 sentence response.

**回答**

以 forward 的全部 GPU kernel 时间 131.743 ms 为分母，非 GEMM kernel 合计占 35.89%；其中普通 `elementwise_kernel` 为 21.941 ms（16.65%），`vectorized_elementwise_kernel` 为 19.915 ms（15.12%），`reduce_kernel` 为 5.066 ms（3.85%）。它们主要来自 residual 与逐元素乘除、RMSNorm、SiLU、mask、softmax 的 max/sum reduction 和 exponential；虽然 FLOPs 少，但需要多次 kernel launch 和显存读写。

### 2.4 问题 (d)

> Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
>
> **Deliverable:** A 1-2 sentence response.

**回答**

Forward 阶段中 GEMM 占 GPU kernel 时间的 64.11%，加入 loss、backward 和 AdamW 后降至 53.08%；optimizer 的 67.404 ms 全部来自逐元素 kernel，backward 也增加了大量逐元素和 reduction kernel，因此非矩阵乘占比上升。该 profile 没有独立运行 `torch.inference_mode()`，这里以 full step 内的 forward 阶段近似 inference kernel 构成；当前模型没有 Dropout 或 BatchNorm，但 autograd 状态仍与真正 inference 不同。

### 2.5 问题 (e)

> Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?
>
> **Deliverable:** A 1-2 sentence response.

**回答**

每层 attention 的两次矩阵乘共调用 24 个 `maxwell_sgemm_128x64_nn` kernel，累计 9.200 ms；手写 softmax 的 max、减法、exp、sum 和除法各调用 12 次，合计约 16.487 ms，反而是矩阵乘的 1.79 倍。两次 attention 矩阵乘约为 1.61 GFLOPs，而按每个 score 元素约 5 个标量操作粗算 softmax 只有约 0.031 GFLOPs，前者约高 51 倍；softmax 仍更慢，说明其 reduction、指数运算、多次 kernel launch 和中间张量显存流量没有获得 GEMM 那样高的计算效率。

---

## 3. 关键证据

### 3.1 Forward kernel 排名

| 排名 | Kernel | 调用次数 | 累计时间 | Forward GPU kernel 占比 |
|---:|---|---:|---:|---:|
| 1 | `sgemm_128x128x8_NT_vec` | 85 | 75.263 ms | 57.13% |
| 2 | `elementwise_kernel` | 278 | 21.941 ms | 16.65% |
| 3 | `vectorized_elementwise_kernel` | 231 | 19.915 ms | 15.12% |
| 4 | `maxwell_sgemm_128x64_nn` | 24 | 9.200 ms | 6.98% |
| 5 | `reduce_kernel` | 49 | 5.066 ms | 3.85% |

### 3.2 如何阅读 `sgemm_128x128x8_NT_vec`

这是 NVIDIA cuBLAS 选择的内部 kernel 短名称，不是 PyTorch API 名称，也不是稳定的公开接口。它可以按以下方式理解：

| 名称部分 | 可推断的信息 | 不能过度推断的内容 |
|---|---|---|
| `sgemm` | `S` 表示 single precision，即 FP32；`GEMM` 表示通用矩阵乘，计算形式类似 $C=\alpha\,\mathrm{op}(A)\mathrm{op}(B)+\beta C$ | 仅凭名称不能确定它来自 Q/K/V、FFN 还是 LM head |
| `128x128x8` | 通常表示该实现围绕约 $128\times128$ 的输出 tile，并沿归约维度以 8 为一个内部步长组织计算 | 它不是完整矩阵的 shape，也不是 CUDA `gridDim`、`blockDim` 或线程数；这是内部实现命名惯例，不应视为稳定 ABI |
| `NT` | 在 cuBLAS 内部约定中，第一操作数采用 non-transposed，第二操作数采用 transposed 变体 | PyTorch 张量采用行主序语义，调用 cuBLAS 时可能交换操作数或转置标志，因此不能直接断言源码一定显式执行了 `A @ B.T` |
| `_vec` | 表示该变体采用了某种 vectorized load/store 数据访问路径 | 名称没有说明具体向量宽度、对齐条件或实际显存吞吐率 |

该 profile 还显示，此 kernel 实际使用 `block=(256,1,1)`、每线程 128 个寄存器和 16,912 bytes 静态 shared memory；同一个名称对应三种 launch grid：

| Grid | 调用次数 | 累计时间 |
|---|---:|---:|
| `(4,3,4)` | 60 | 37.067 ms |
| `(4,12,4)` | 24 | 33.655 ms |
| `(2,79,4)` | 1 | 4.541 ms |

这三组调用合计 85 次和 75.263 ms。不同 grid 共用同一个 kernel 名称，说明名称描述的是 cuBLAS 实现变体，而不是具体矩阵尺寸；85 次的层级归因来自 [TransformerLM](../../assignment1-basics/cs336_basics/model.py#L196-L235) 的结构：每层有 Q、K、V、attention output 四个线性层和 SwiGLU 的三个线性层，共 $7\times12=84$ 次，最终 LM head 再增加 1 次。

在 GTX 1060 上，这个 `sgemm` 是普通 FP32 CUDA-core GEMM，不是 Tensor Core kernel，因为 Pascal GP106 本身没有 Tensor Core。换用不同 GPU、CUDA/cuBLAS 版本、dtype 或矩阵 shape 后，cuBLAS 可能选择完全不同的 kernel 名称。

### 3.3 为什么 backward 的最高耗时 kernel 发生变化

数学上以列向量记线性层为 $y=Wx$；在 PyTorch 中特征位于最后一维，展平 batch 和 sequence 后对应 $Y=XW^\top$。反向传播需要两个矩阵乘：

- 输入梯度：$dX=dY\,W$；
- 权重梯度：$dW=dY^\top X$。

因此，一个 forward 线性层只有一次 GEMM，而 backward 要为同一线性层执行两次方向不同的 GEMM。该 profile 的数据为：

| 阶段 | Kernel | 调用次数 | 累计时间 | 单次平均时间 |
|---|---|---:|---:|---:|
| forward | `sgemm_128x128x8_NT_vec` | 85 | 75.263 ms | 885.450 µs |
| backward | `sgemm_128x128x8_NN_vec` | 85 | 80.300 ms | 944.700 µs |
| backward | `sgemm_128x128x8_TN_vec` | 85 | 73.354 ms | 862.989 µs |

Backward 的两类 kernel 各出现 85 次，与 85 个线性层 forward GEMM 一一对应。这里的 85 是按完全相同的 kernel 名称分组后的次数，不是 backward 的 GEMM 总数：每个线性层分别产生一个输入梯度 GEMM 和一个权重梯度 GEMM，170 次调用被拆为 `NN` 85 次和 `TN` 85 次。

同样，每个 self-attention 层在 forward 中包含两次矩阵乘：一次计算 attention score $QK^\top$，一次计算 softmax 后的权重与 value 的乘积 $PV$。`small` 模型有 12 层，因此 forward 调用数是 $12\times2=24$；每个矩阵乘的 backward 又分别计算两个输入梯度，所以 backward 调用数是 $12\times2\times2=48$，在 profile 中拆成 24 次 `maxwell_sgemm_128x64_tn` 和 24 次 `maxwell_sgemm_128x64_nt`。

| GEMM 来源 | Forward 调用次数 | Backward 调用次数 |
|---|---:|---:|
| 线性层 | 85 | 170 |
| Self-attention | 24 | 48 |
| 合计 | 109 | 218 |

因此该 profile 中 backward 的 GEMM 调用总数确实恰好是 forward 的两倍；最高耗时的单个 kernel 仍然只有 85 次，是因为两个反向 GEMM 使用了不同的 kernel 变体，统计时落入两个独立分组。`NN` 变体的累计时间超过 forward 的 `NT`，一方面是因为反向矩阵的方向和访存布局不同，另一方面是 cuBLAS 针对这些 shape 选择的 kernel 效率不同；但由于这些名称属于 cuBLAS 内部实现，不能仅凭 `NN` 或 `TN` 判断它一定对应 $dX$ 还是 $dW$，精确映射还需要结合 CUDA API 参数和 correlation ID。

### 3.4 Batch-head 维度的来源

#### 3.4.1 Batch 与 head 不是同时出现的概念

Batch 维度早于 Transformer 和多头注意力。神经网络训练通常把多个样本组成 mini-batch，一次执行相同的算子，以提高硬件利用率并估计批次梯度；在语言模型中，输入 token id 的形状通常是 `(B,S)`，嵌入后变为 `(B,S,d_model)`，其中 `B` 是 batch size，`S` 是序列长度。

Head 维度来自 multi-head attention。其目标是让多个注意力头在不同表示子空间中独立计算注意力，而不是让一个注意力矩阵承担全部关系。模型将总特征维度拆成 $H$ 个 head，每个 head 的维度为 $d_h=d_{\text{model}}/H$。

因此：

- batch 不是为了 MHA 才出现，它是通用的批处理维度；
- head 是 MHA 引入的结构维度；
- `batch-head` 不是模型中新定义的一种语义维度，而是实现 batched matrix multiplication 时对两个前导维度的合称。

#### 3.4.2 当前代码中何时出现 head 维度

进入 attention 前，输入 `x` 和 Q、K、V 投影的形状都是：

```text
(B, S, d_model)
```

在 [`split_heads`](../../assignment1-basics/cs336_basics/model.py#L150-L157) 中，代码先 reshape，再 transpose：

```python
return t.reshape(*batch, seq, self.num_heads, self.head_dim).transpose(-3, -2)
```

形状变化为：

$$ (B,S,d_{\text{model}}) \rightarrow (B,S,H,d_h) \rightarrow (B,H,S,d_h) $$

head 轴正是在这里显式出现。Attention 计算结束后，[输出](../../assignment1-basics/cs336_basics/model.py#L162-L164) 再从 `(B,H,S,d_h)` transpose 并 reshape 回 `(B,S,d_model)`，所以 head 维度只存在于多头注意力内部。

#### 3.4.3 为什么称为 batch-head

`torch.matmul` 把矩阵最后两个维度用于矩阵乘，把此前的所有维度视为 batch dimensions。于是：

$$ QK^\top:(B,H,S,d_h)@(B,H,d_h,S)\rightarrow(B,H,S,S) $$

这里 `(B,H)` 都是矩阵乘的 batch dimensions。实现上可以把它们理解为合并后的逻辑批次 `B×H`，即同时计算 `B×H` 组互相独立的小矩阵乘，所以常简称为 batch-head。

当前配置中 `B=2`、`H=12`，每个 attention 矩阵乘节点包含 24 个逻辑矩阵乘。每层有 $QK^\top$ 和 $PV$ 两个节点，12 层共有 $12\times2\times24=576$ 个逻辑矩阵乘，但这不等于 576 次 CUDA kernel launch。

#### 3.4.4 为什么 profile 中 forward 仍只有 24 次 kernel launch

如果使用 Python 循环逐个处理 batch 和 head，确实可能产生大量独立 kernel launch；PyTorch 会把 `(B,H)` 交给 batched GEMM，一次算子调用批量处理所有 batch-head 矩阵。本 profile 中，每层的 $QK^\top$ 和 $PV$ 各对应一次 `maxwell_sgemm_128x64_nn` 启动，因此 CUDA kernel 启动数是：

$$ 12\text{ layers}\times2\text{ attention matmul nodes}=24\text{ launches} $$

也就是说：

| 计数对象 | Forward 数量 |
|---|---:|
| Attention matmul 节点 | 24 |
| 每个节点内的 batch-head 矩阵 | 24 |
| 逻辑小矩阵乘总数 | 576 |
| 本 profile 中的 CUDA kernel launch | 24 |

把 batch-head 合并交给 batched GEMM 的背景是减少 Python 循环和 kernel launch 开销，并让 GPU 同时获得足够多的独立矩阵工作。它是向量化实现策略，不是 MHA 数学定义要求必须把两个维度物理合并；不同 PyTorch、cuBLAS 版本或输入 shape 也可能选择不同数量的底层 kernel。

### 3.5 Softmax 分解

当前 [softmax 实现](../../assignment1-basics/cs336_basics/nn_utils.py#L7-L11) 没有融合，因此每层至少启动 max、减法、exp、sum 和除法五类 kernel：

| 操作 | 调用次数 | 累计时间 |
|---|---:|---:|
| max reduction | 12 | 2.299 ms |
| 减去行最大值 | 12 | 3.996 ms |
| exponential | 12 | 3.980 ms |
| sum reduction | 12 | 1.992 ms |
| 除以行和 | 12 | 4.220 ms |
| 合计 | 60 | 16.487 ms |

这里根据 demangled kernel 名称、tensor 对应的 grid shape 以及“每层各调用一次”的计数完成归因。由于原 profile 没有 self-attention 子阶段 NVTX range，这一归因属于有代码结构佐证的离线推断，而不是 GUI 中直接按 attention range 过滤得到的结果。

---

## 4. 结论边界

1. 本报告严格只使用一个 `small/B=2/S=512/full` profile。
2. 它不能回答不同模型尺寸或不同 context length 的扩展趋势。
3. `forward` 是 full step 中开启 autograd 的训练 forward，不是单独的 inference-mode profile。
4. Softmax 子阶段由 kernel 名称、shape 和调用次数推断；增加细粒度 NVTX range 后可直接验证。
5. GTX 1060 没有 Tensor Core，结论不能外推到课程常用的 A100、H100 或 B200。
