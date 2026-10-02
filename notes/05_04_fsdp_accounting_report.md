# FSDP Accounting 实验报告

> 状态：静态 `xl` 内存核算、FSDP accounting 工具和 2-rank CPU/Gloo 调度实验已完成。当前环境没有两张可见 CUDA GPU，无法生成 handout 要求的 2-GPU/NCCL/`xl` timings 与 Nsight 截图；本文不将 CPU 证据冒充 GPU overlap 证据。

## 1. Handout 直接答案

### 1.1 (a) Peak memory 预期

忽略 all-gather 预分配 buffer，标准 FP32 `xl` 模型在 2-rank optimizer-state sharding 后的 persistent state 峰值约为 `38.074 GiB/rank`；FSDP 将 Linear/Embedding 的参数、梯度和 moments 都分片，只复制 65 个 RMSNorm 参数张量，persistent state 约为 `25.384 GiB/rank`。因此 FSDP 相对上一节再节省 `12.690 GiB/rank`，即 33.33%；相对普通 DDP 的 `50.765 GiB/rank` 则节省约 50.00%。

### 1.2 (b) All-gather 是否及时

正式 2-GPU/NCCL/`xl` 未测，因此不能声称 all-gather 能或不能及时完成，也没有提交伪造的 Nsight 截图。2-rank CPU/Gloo 小模型 smoke 中，FP32 forward 有 65.00% 的 gather 在 layer 使用前已完成，逐 step 仍暴露 `1.893 ± 0.291 ms` wait；这是调度链路证据，不是 CUDA stream overlap 结论。

## 2. 背景：什么叫“通信及时”

### 2.1 Prefetch 窗口

对第 $i$ 层，当前实现在线性顺序的第 $i-2$ 层 forward 完成后启动 all-gather。设：

- $C_i$：层 $i$ weight all-gather 完成时间；
- $F_{i-1}$：中间一层计算时间；
- $W_i$：层 $i$ pre-hook 中剩余等待。

理想近似为：

$$ W_i=\max(0,C_i-F_{i-1}). $$

若 $W_i=0$，通信在 weight 使用前完成；若 $W_i>0$，这部分 wait 直接进入 forward critical path。

### 2.2 Host ready 不等于 GPU timeline overlap

CPU/Gloo 下，`Work.is_completed()` 在 wait 前为真，可以作为 host-observed ready。CUDA/NCCL 下，即使 Python 调用已经返回，NCCL kernel 也可能仍在设备通信 stream 上执行；`Work.wait()` 还可能只是向当前 CUDA stream 插入依赖。

因此 GPU 上必须同时满足：

1. Nsight 显示 `ncclKernel_*` 在目标 layer compute 前结束；
2. `fsdp_all_gather_wait:*` 附近没有形成显著 GPU idle gap；
3. all-gather 与前一层 GEMM 有真实时间重叠；
4. 完整 step time 没有被资源争用反向拖慢。

### 2.3 Mixed-precision payload

设 shardable FP32 weight 总量为 $M_s$。每个 forward 要 materialize 一次完整 weight：

- FP32 reconstructed payload 为 $M_s$；
- FP16 reconstructed payload 为 $M_s/2$。

这里的 reconstructed payload 是所有 full tensor bytes 之和，不等于物理链路流量。ring all-gather 的每 rank 物理 send/receive volume 近似为：

$$ V_{\mathrm{all\text{-}gather}}\approx\frac{N-1}{N}M_s. $$

## 3. `xl` 静态内存核算

### 3.1 模型组成

标准配置：

| 项目 | 值 |
|---|---:|
| $d_{\mathrm{model}}$ | 2560 |
| $d_{\mathrm{ff}}$ | 10240 |
| layers | 32 |
| heads | 32 |
| vocabulary | 10,000 |
| 参数量 | 3,406,809,600 |
| 参数张量 | 291 |
| shardable Linear/Embedding 张量 | 226 |
| replicated RMSNorm 张量 | 65 |

总 FP32 parameter 为 `13,627,238,400 B = 12.691 GiB`。其中 shardable weight 为 `12.691 GiB`，replicated norms 只有 `0.635 MiB`。

### 3.2 每 rank persistent state

| 组成 | Optimizer state sharding | FSDP |
|---|---:|---:|
| parameter | 12.691 GiB | 6.346 GiB |
| gradient | 12.691 GiB | 6.346 GiB |
| AdamW moments | 12.691 GiB | 12.692 GiB |
| 总计 | 38.074 GiB | 25.384 GiB |

FSDP 的 moment 项略高于 `12.691 GiB`，原因是 RMSNorm 参数保持复制，以及每个 shardable weight 独立向上取整产生 padding。

精确结果：

```text
optimizer-state-sharding rank-max: 40,881,725,440 B
FSDP per rank:                       27,255,808,000 B
savings:                             13,625,917,440 B
savings ratio:                       33.3301%
```

相对普通 DDP：

```text
replicated DDP persistent state: 54,508,953,600 B/rank
FSDP persistent state:           27,255,808,000 B/rank
savings ratio:                   49.9976%
```

### 3.3 为什么不是严格 50%

本实现没有分片 RMSNorm，并对每个 weight 独立 padding：

$$ M_{\mathrm{FSDP}}=4\left(\sum_l\left\lceil\frac{P_l}{N}\right\rceil b+M_{\mathrm{replicated}}\right). $$

其中 $P_l$ 是第 $l$ 个 shardable weight 元素数，$b=4$ B。若忽略 norms 和 padding，2-rank 下才简化为 $4M/2=2M$。

### 3.4 “忽略 all-gather buffer”的含义

以上只计算长期驻留的 parameter shard、gradient shard 和 AdamW moments。实际 peak 还可能包括：

1. 当前层完整 materialized weight；
2. 两层 lookahead 的 prefetched full weights；
3. collective staging buffer；
4. activation 与 saved tensors；
5. CUDA context、workspace 和 allocator fragmentation。

所以 `25.384 GiB/rank` 是题目允许忽略 gather buffer 后的静态结果，不是预计的 `torch.cuda.max_memory_allocated()` 精确值。

## 4. Accounting 实现

### 4.1 静态核算

[`build_fsdp_static_accounting`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_accounting.py#L151-L221) 在 meta device 上构造真实 `TransformerLM`，按实际模块类型区分 shardable weight 与复制参数，并复用逐 weight padding 规则。

### 4.2 动态记录

[`FSDPAccountingConfig`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_accounting.py#L33-L75) 验证 backend、world size、模型和硬件前置条件。

核心 FSDP 不持有计时数组，也不提供实验结果查询方法。它只通过 [`FSDPAllGatherEvent`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_observer.py#L13-L29) 发出 collective 生命周期事件；[`FSDPCommunicationRecorder`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_accounting.py#L78-L148) 作为 accounting adapter 独立完成计时、`Work.is_completed()` 检查、通信字节计算和 NVTX wait 标记。

[`_run_step`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_accounting.py#L277-L312) 记录：

- rank-max step time；
- rank-max forward time；
- 每个 phase 的 all-gather records；
- 使用前 wait；
- `Work.is_completed()` 状态；
- communicated dtype 和 reconstructed bytes。

[`benchmark_fsdp.py`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/benchmark_fsdp.py) 让每个 dtype/repeat 使用隔离进程，并随机化顺序。

## 5. CPU/Gloo 调度实验

### 5.1 设置

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon Platinum 8336C @ 2.30 GHz |
| backend / ranks | Gloo / 2 |
| 模型 | $d_{\mathrm{model}}=64$、$d_{\mathrm{ff}}=128$、2 layers、4 heads |
| vocabulary / context | 256 / 32 |
| global/local batch | 4 / 2 |
| shardable weight tensors | 16 |
| warmup / measurement | 3 / 10 |
| repeats | 3 个隔离进程 |
| CPU threads | 每 worker 1 |

正式计时前的初始化 step 和 warmup 会先建立并验证真实 forward/backward 执行顺序，因此测量样本不包含首轮顺序发现开销。

命令：

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python scripts/benchmark_fsdp.py \
  --backend gloo \
  --world-size 2 \
  --compute-dtypes float32 float16 \
  --model-size xl \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --num-heads 4 \
  --vocab-size 256 \
  --global-batch-size 4 \
  --context-length 32 \
  --warmup-steps 3 \
  --measurement-steps 10 \
  --repeats 3 \
  --num-threads 1 \
  --output-dir benchmark_results/fsdp/cpu_gloo_decoupled_final_20261003
```

原始汇总见 [`cpu_gloo_summary.json`](./assets/fsdp/cpu_gloo_summary.json)，环境和 hash 见 [`provenance.json`](./assets/fsdp/provenance.json)。

### 5.2 结果

| compute dtype | step mean ± repeat std | forward mean ± repeat std | forward exposed wait | forward ready fraction | forward reconstructed bytes |
|---|---:|---:|---:|---:|---:|
| FP32 | 23.519 ± 1.037 ms | 7.654 ± 0.465 ms | 1.893 ± 0.291 ms | 65.00% | 448 KiB |
| FP16 | 39.885 ± 1.298 ms | 12.987 ± 0.183 ms | 2.629 ± 0.143 ms | 73.13% | 224 KiB |

每 step 的 forward 和 backward 各执行 16 次 weight all-gather。FP16 reconstructed bytes 恰好减半，证明 weight 在通信前完成了 cast。

### 5.3 解释

FP32 下仍有 35.00% 的 forward gather 在 layer 使用前未完成，累计 exposed wait 约占 forward 的 24.73%，因此在 CPU/Gloo 条件下答案是“没有全部及时完成”。

FP16 虽然 payload 减半，但 CPU FP16 matmul/cast 效率差，step 比 FP32 更慢；这不否定 GPU mixed precision。通信字节、计算吞吐和 overlap 必须分别观察。

`ready_before_use_fraction` 来自 Gloo `Work.is_completed()`，只能证明 host-observed 状态。它不能回答 NCCL kernel 是否与 GPU GEMM 重叠。

## 6. 正式 NCCL/Nsight 方法

### 6.1 运行命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python scripts/benchmark_fsdp.py \
  --backend nccl \
  --world-size 2 \
  --compute-dtypes float16 \
  --model-size xl \
  --vocab-size 10000 \
  --global-batch-size 4 \
  --context-length 512 \
  --warmup-steps 3 \
  --measurement-steps 10 \
  --repeats 3 \
  --output-dir benchmark_results/fsdp/nccl_xl_fp16
```

当前环境执行时 fail-fast：

```text
RuntimeError: NCCL requires 2 distinct visible CUDA devices
```

### 6.2 Nsight 命令

accounting recorder 为 all-gather 的 exposed wait 添加 NVTX range；NCCL collective 与 GEMM kernel 由 Nsight 直接采集。具备两张足够显存 GPU 时运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=benchmark_results/fsdp/nsys_xl_fp16 \
  uv run python scripts/benchmark_fsdp.py \
    --backend nccl \
    --world-size 2 \
    --compute-dtypes float16 \
    --model-size xl \
    --vocab-size 10000 \
    --global-batch-size 4 \
    --context-length 512 \
    --warmup-steps 1 \
    --measurement-steps 2 \
    --repeats 1 \
    --output-dir benchmark_results/fsdp/nsys_xl_fp16_data
```

应截图展示：

1. 对应 NCCL all-gather kernel；
2. 前一层 GEMM；
3. `fsdp_all_gather_wait:<module>`；
4. 当前层 GEMM。

若 NCCL kernel 在 wait range 前结束，并与前一层 GEMM 重叠，则该层通信及时；否则应测量 wait 到当前层首个 GEMM 的 gap。

## 7. 结论与边界

1. 忽略 gather buffer 时，2-rank FSDP 把 `xl` persistent state 从 optimizer sharding 的 38.074 GiB/rank 降至 25.384 GiB/rank。
2. 2-rank CPU/Gloo 显示 prefetch 有效果，但没有隐藏全部 forward all-gather。
3. FP16 weight communication payload减半；CPU FP16 性能不代表 GPU。
4. 正式 2-GPU/NCCL `xl` timings 与 Nsight 截图未获得，不能对 GPU 上“是否及时”下结论。
5. 当前逐 weight collective 的 latency 明显，正式 FSDP 应使用 flat bucket 与更成熟的 stream-aware prefetch。

## 8. 参考资料

1. PyTorch, [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/stable/fsdp.html).
2. Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277).
3. Rajbhandari et al., [ZeRO 原始论文](./references/distributed_training/zero_memory_optimizations_toward_training_trillion_parameter_models.pdf).
4. NVIDIA, [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
