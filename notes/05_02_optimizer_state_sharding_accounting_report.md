# Optimizer State Sharding Accounting 实验报告

> 状态：accounting 工具、2-rank CPU/Gloo 实测和 `xl` 静态核算已完成。当前节点没有可见 CUDA device，正式 1-node × 2-GPU/NCCL/`xl` peak-memory 与迭代时间无法实测；本文明确保留该证据边界，不用 CPU 数值冒充 GPU 结果。

## 1. 问题与直接答案

### 1.1 (a) 内存

当前环境无法给出 handout 指定的双 GPU allocator 实测值。对标准 FP32 `xl` 模型进行 meta-device 精确参数核算后，每 rank 的持久 parameter、gradient 和 AdamW moment 下界如下：

| 检查点 | 普通 AdamW | 2-rank ShardedOptimizer |
|---|---:|---:|
| 模型初始化后 | 12.691 GiB | 12.691 GiB |
| optimizer step 前 | 25.383 GiB | 25.383 GiB |
| 第一次 optimizer step 后 | 50.765 GiB | 38.074 GiB |

结果符合预期：AdamW 状态在第一次 `step()` 中惰性创建，分片前后在前两个检查点没有 optimizer-state 差异；step 后每 rank 的 moment 从 25.383 GiB 降至约 12.691 GiB，使持久 parameter + gradient + moment 总量减少约 25%。这些数字不包含 activation、DDP 临时通信 buffer、CUDA context、allocator reserved memory 和算子 workspace，因此是静态下界，不是 GPU peak。

### 1.2 (b) 速度

正式 2-GPU/NCCL/`xl` 时间未测。本机 2-rank CPU/Gloo 缩小模型中，普通 AdamW 为 `8.493 ± 0.135 ms/step`，分片版本为 `11.626 ± 0.581 ms/step`，分片版本慢 36.89%；optimizer 阶段从 `1.539 ± 0.025 ms` 增至 `4.298 ± 0.253 ms`。

该 CPU 结果只验证当前实现会因更新后的逐参数 broadcast 产生额外开销，不能外推为 GPU/NCCL/`xl` 的性能结论。

### 1.3 (c) 与 ZeRO Stage 1 的差异

两者都只让一个 rank 保存并更新每个参数对应的 optimizer state，因而 optimizer-state memory 均近似降至普通 DDP 的 $1/N$。本实现先由既有 DDP 完成完整 gradient all-reduce，再逐参数 broadcast 更新后的参数，按 ZeRO 论文的带宽计数约为 $3\Psi$，而 ZeRO Stage 1 可将 gradient reduce-scatter 与 updated-parameter all-gather 组织为约 $2\Psi$，不高于标准 DDP all-reduce。

此外，本实现按参数张量贪心分配、发起 $K$ 次 broadcast，并使用 rank-local checkpoint；ZeRO 工业实现通常使用 flat partition、bucket collective、通信重叠和 reshard 支持。

## 2. Accounting 口径

### 2.1 三个检查点

脚本记录首个训练 step 的三个位置：

1. `after_model_initialization`：模型、DDP wrapper 和 optimizer 已创建，输入尚未分配；AdamW state 尚未惰性初始化。
2. `before_optimizer_step`：forward、backward 和 DDP gradient synchronization 已完成；完整 gradient 仍然驻留。
3. `after_optimizer_step`：AdamW 已完成首次更新，`exp_avg`、`exp_avg_sq` 和每参数 `step` 已创建。

首个 step 专门用于 memory profiling，不进入稳态 timing。随后执行 warmup，再记录 measurement steps。

### 2.2 四种内存指标

| 指标 | 含义 | 适用范围 |
|---|---|---|
| `parameter/gradient/state tensor bytes` | 对活跃 Tensor 执行 `numel() * element_size()` | 精确解释训练状态组成 |
| CPU current RSS | `/proc/self/statm` 的当前 resident pages | 包含 runtime、allocator 和共享库 |
| CPU max RSS | `getrusage().ru_maxrss` | 进程启动以来单调不减的峰值 |
| CUDA allocated/reserved/phase peak | PyTorch CUDA allocator 指标 | 正式 GPU peak-memory |

Tensor-byte accounting 不包含 activation、allocator metadata 和临时 workspace；RSS 又无法把 Python runtime 与 Tensor 分开。两者必须并列报告，不能互相替代。

CUDA 模式在每个阶段开始前调用 `torch.cuda.reset_peak_memory_stats()`，检查点同时保存当前 allocated/reserved 和该阶段 peak allocated/reserved。

### 2.3 AdamW 状态

设 FP32 参数总大小为 $M$。忽略很小的每参数 `step` 标量：

$$ M_{\text{baseline,persistent}}=M_{\theta}+M_g+M_m+M_v=4M. $$

2-rank 理想均衡分片后：

$$ M_{\text{sharded,persistent}}=M_{\theta}+M_g+\frac{M_m+M_v}{2}=3M. $$

因此 optimizer state 自身减少 50%，但这部分只占 parameter + gradient + moments 总量的一半，所以持久训练状态减少 25%。

## 3. 实现

### 3.1 Case 配置与 fail-fast

[`OptimizerShardingAccountingConfig`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/optimizer_sharding_accounting.py#L33-L80) 固定：

- optimizer variant：`baseline` 或 `sharded`；
- 相同 DDP variant；
- backend、world size 和模型维度；
- global/local batch；
- warmup、measurement steps 和线程数。

NCCL 模式在 spawn 前检查可见 GPU 数量，避免 worker 启动后才 hang。

### 3.2 静态核算

[`build_static_accounting`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/optimizer_sharding_accounting.py#L103-L171) 在 `meta` device 构造真实 `TransformerLM`：

1. 从真实参数 shape 计算 parameter count 和 FP32 bytes；
2. 复用 `ShardedOptimizer` 的按参数字节 greedy owner 规则；
3. 分别计算每 rank 的 owner parameter bytes 和 AdamW moment bytes；
4. 不申请对应 CPU/GPU storage。

这种方式比手写 Transformer 参数量公式更不容易漏掉 embedding、final norm 或 lm head，但结果仍是静态下界。

### 3.3 阶段快照

[`_memory_snapshot`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/optimizer_sharding_accounting.py#L202-L245) 同时记录：

- 完整 parameter 与当前 gradient bytes；
- optimizer state tensor 按 device 的 bytes；
- owner parameter bytes；
- RSS；
- CUDA current 和 phase peak。

[`_run_first_profiled_step`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/optimizer_sharding_accounting.py#L287-L319) 在第一次 optimizer step 前后重置并读取 CUDA peak，捕获 AdamW state 的首次创建。

### 3.4 Timing

[`_run_timed_step`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/optimizer_sharding_accounting.py#L322-L356) 记录：

1. 完整 `zero_grad -> forward -> loss -> backward -> gradient sync -> optimizer` step；
2. gradient sync；
3. optimizer step，其中 sharded case 包含 parameter broadcast。

每个 sample 在计时窗外 barrier，并在 CUDA 上同步设备。分布式结果先逐 sample 取最慢 rank，再计算 mean、population standard deviation、median 和 p95。

### 3.5 隔离与重复

[`benchmark_optimizer_sharding.py`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/benchmark_optimizer_sharding.py) 为每个 variant/repeat 重新 spawn worker，随机化 baseline/sharded 顺序，避免模型、allocator cache 和 optimizer state 在 case 之间复用。

原始 CPU suite 汇总见 [`cpu_gloo_summary.json`](./assets/optimizer_sharding_accounting/cpu_gloo_summary.json)，环境与校验信息见 [`provenance.json`](./assets/optimizer_sharding_accounting/provenance.json)。

## 4. 实验设置

### 4.1 正式配置

handout 要求：

| 项目 | 值 |
|---|---|
| 节点 | 1 |
| GPU/ranks | 2 |
| backend | NCCL |
| 模型 | `xl` |
| $d_{\text{model}}$ | 2560 |
| $d_{\text{ff}}$ | 10240 |
| layers | 32 |
| heads | 32 |
| vocabulary | 10,000 |
| context length | 512 |
| global/local batch | 4 / 2 |
| dtype | FP32 |
| optimizer | `torch.optim.AdamW(foreach=False)` |
| DDP | flat-gradient，baseline/sharded 相同 |

正式复跑命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python scripts/benchmark_optimizer_sharding.py \
  --backend nccl \
  --world-size 2 \
  --ddp-variant flat \
  --model-size xl \
  --vocab-size 10000 \
  --global-batch-size 4 \
  --context-length 512 \
  --warmup-steps 3 \
  --measurement-steps 10 \
  --repeats 3 \
  --num-threads 1 \
  --output-dir benchmark_results/optimizer_sharding_accounting/nccl_xl
```

当前环境执行时得到：

```text
RuntimeError: NCCL requires 2 distinct visible CUDA devices
```

当前 `torch.cuda.device_count()` 为 0；已知 6 GiB GTX 1060 即使可见，也小于 `xl` 单份 FP32 参数的 12.691 GiB，无法完成模型初始化。

### 4.2 CPU/Gloo smoke

为验证完整测量链路，本机使用：

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon Platinum 8336C @ 2.30 GHz |
| backend / ranks | Gloo / 2 |
| 模型标签 | `xl`，但显式覆盖为 $d_{\text{model}}=64$、$d_{\text{ff}}=128$、2 layers、4 heads |
| vocabulary / context | 256 / 32 |
| global/local batch | 4 / 2 |
| parameter count | 115,008 |
| parameter tensors | 21 |
| 参数存储 | 460,032 B |
| optimizer | PyTorch AdamW，`foreach=False` |
| warmup / measurement | 3 / 10 |
| repeats | 3 个隔离进程，随机顺序 |
| CPU threads | 每 worker 1 |

## 5. (a) 内存结果

### 5.1 `xl` 静态下界

`xl` 含 3,406,809,600 个参数、291 个参数张量。FP32 单份参数为：

$$ M=3{,}406{,}809{,}600\times4\text{ B}=13{,}627{,}238{,}400\text{ B}=12.691\text{ GiB}. $$

| 组成 | 普通 AdamW/rank | Sharded rank 0 | Sharded rank 1 |
|---|---:|---:|---:|
| 完整 parameter | 12.691 GiB | 12.691 GiB | 12.691 GiB |
| 完整 gradient | 12.691 GiB | 12.691 GiB | 12.691 GiB |
| AdamW moments | 25.383 GiB | 12.691 GiB | 12.691 GiB |
| 合计 | 50.765 GiB | 38.074 GiB | 38.074 GiB |
| owner parameter tensors | 291 | 97 | 194 |

rank 0/1 分别拥有 6,813,614,080 B 和 6,813,624,320 B 参数，差异只有 10,240 B。参数张量数虽然是 97 对 194，但大矩阵字节达到近似严格均衡，说明按字节分配比按 tensor 数量轮转更符合 optimizer-state 内存目标。

每参数 `step` 标量总计只有 1,164 B；默认非 capturable AdamW 通常将它们留在 CPU，因此表中 GPU 持久下界只计算两个 moments。

### 5.2 CPU/Gloo 精确 Tensor bytes

下表采用三个 repeat 的逐次 rank-max，再取均值；Tensor bytes 每次完全一致：

| 检查点 | variant | parameter | gradient | optimizer state | 已核算合计 |
|---|---|---:|---:|---:|---:|
| 模型初始化后 | baseline | 449.25 KiB | 0 | 0 | 449.25 KiB |
| 模型初始化后 | sharded | 449.25 KiB | 0 | 0 | 449.25 KiB |
| step 前 | baseline | 449.25 KiB | 449.25 KiB | 0 | 898.50 KiB |
| step 前 | sharded | 449.25 KiB | 449.25 KiB | 0 | 898.50 KiB |
| step 后 | baseline | 449.25 KiB | 449.25 KiB | 898.58 KiB | 1,797.08 KiB |
| step 后 | sharded rank-max | 449.25 KiB | 449.25 KiB | 481.55 KiB | 1,380.05 KiB |

小模型不能按参数粒度完美二分：rank 0/1 分别拥有 213,504 B 和 246,528 B 参数，对应 AdamW state 为 427,040 B 和 493,108 B。rank-max state 相比 baseline 减少 46.41%，而两个 rank 的 state 总和仍恰好等于一份 baseline state。

### 5.3 CPU RSS

| 检查点 | baseline rank-max mean | sharded rank-max mean |
|---|---:|---:|
| 模型初始化后 | 599.31 MiB | 597.40 MiB |
| step 前 | 610.48 MiB | 608.61 MiB |
| step 后 | 610.96 MiB | 608.99 MiB |

RSS 在模型初始化后已经相差约 1.92 MiB，大于本实验 rank-max optimizer state 的约 0.41 MiB 差值，说明 Python runtime、进程映射和 CPU allocator 波动掩盖了小模型状态差异。这里不把约 1.97 MiB 的 step 后 RSS 差直接归因于 sharding；精确 tensor accounting 才是状态节省证据。

## 6. (b) 速度结果

| variant | step mean ± repeat std | optimizer mean ± repeat std | 相对 baseline |
|---|---:|---:|---:|
| baseline | 8.493 ± 0.135 ms | 1.539 ± 0.025 ms | $1.000\times$ |
| sharded | 11.626 ± 0.581 ms | 4.298 ± 0.253 ms | $1.369\times$ |

当前 CPU/Gloo 小模型上，sharded step 慢 36.89%，optimizer 阶段约为 baseline 的 2.79 倍。两者使用完全相同的一次 flat-gradient all-reduce；差异主要来自 sharded optimizer 在本地更新后额外执行 21 次 parameter broadcast，小消息 latency 在该配置中占主导。

这个结果不能预测 `xl` GPU 的绝对时间：大参数提高带宽占比，NCCL 与 GPU optimizer kernel 的成本结构也不同。不过当前实现逐参数发起 collective，因此即使 payload 不变，291 个参数张量仍会带来明显的启动开销；正式结果必须由上述 NCCL 命令确认。

## 7. (c) 与 ZeRO Stage 1 对比

设全部参数/梯度的通信数据量为 $\Psi$，沿用 ZeRO 论文只比较带宽项的记法。

标准 ring all-reduce 可分成：

```text
gradient reduce-scatter: Psi
gradient all-gather:     Psi
total:                   2 Psi
```

当前教学实现执行：

```text
完整 gradient all-reduce:      2 Psi
更新后 parameter synchronization: 约 1 Psi
total:                         约 3 Psi
```

ZeRO Stage 1 则可执行：

```text
gradient reduce-scatter 到 owner: Psi
owner 更新自己的 parameter shard
updated-parameter all-gather:     Psi
total:                            2 Psi
```

因此两者的 optimizer-state 内存目标相同，但当前实现相对标准 DP 增加约 50% 的带宽量；ZeRO Stage 1 用 updated-parameter all-gather 替换 gradient all-gather，保持标准 DP 的通信量。若在 backward 期间进一步只保留 owner gradient 并及时释放其他 gradient storage，则增加的是 gradient partitioning，即 $P_{os+g}$ / ZeRO Stage 2 的内存语义。

实现层面还有以下差异：

| 当前实现 | ZeRO Stage 1 |
|---|---|
| 按完整参数张量分配 owner | 通常 flatten 后按近似等长 shard 分区 |
| 每参数一次 broadcast | bucketed reduce-scatter / all-gather |
| gradient communication 与 optimizer 分离 | 通信计划与参数 shard 更新协同 |
| 不重叠 parameter broadcast | 可进行 bucket 调度和 overlap |
| rank-local state dict，不支持 reshard | 正式实现通常提供分布式 checkpoint 与 reshard |

## 8. 结论与边界

1. 本实现正确将 AdamW state 从每 rank 一份降为所有 rank 合计一份；2-rank 理想情况下 optimizer state 减少 50%，持久 parameter + gradient + moments 减少 25%。
2. 模型初始化后与第一次 optimizer step 前没有 state-memory 差异，因为 AdamW state 是惰性创建的；差异从首次 `step()` 后出现。
3. 当前简单实现以内存换额外 parameter broadcast，在 CPU/Gloo 小模型上产生 36.89% step 开销。
4. `xl` 静态状态已超过 38 GiB/rank，当前无 CUDA 节点无法执行 handout 指定实验；正式 peak 和速度保持未测状态。
5. 本文没有把静态下界、CPU RSS 或 Gloo timing 冒充 CUDA allocator peak 与 NCCL timing。

## 9. 参考资料

1. Rajbhandari et al., [ZeRO 原始论文 PDF](./references/distributed_training/zero_memory_optimizations_toward_training_trillion_parameter_models.pdf).
2. PyTorch, [`torch.cuda.memory.max_memory_allocated`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html).
3. PyTorch, [`ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/stable/distributed.optim.html).
