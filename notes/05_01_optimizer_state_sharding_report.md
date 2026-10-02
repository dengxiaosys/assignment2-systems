# Optimizer State Sharding 实验报告

## 1. 实验问题与结论

### 1.1 问题

普通 Distributed Data Parallel（DDP）在每个 rank 上复制完整的：

1. 模型参数；
2. 参数梯度；
3. optimizer state。

这里的“复制”指每个 rank 的内存中都长期驻留一份完整数据，并不表示三者都会在每个 step 重新通信复制。普通 DDP 通常只在初始化时广播一次模型参数；optimizer state 由各 rank 根据相同梯度独立创建并执行相同更新；只有梯度需要在每次 backward 后通过 all-reduce 同步。正因为每个 rank 都使用相同参数、同步后的相同梯度和相同 optimizer state 执行相同更新，后续参数不需要再逐 step 广播。

本实验实现 `ShardedOptimizer`：每个参数只指定一个 owner rank，只有 owner 上的本地 optimizer 保存该参数的状态并执行更新；更新结束后，owner 将新参数广播给其他 rank。

本实验对应 handout 的 `optimizer_state_sharding`。下一题 `optimizer_state_sharding_accounting` 要求的 `xl` 模型双 GPU peak-memory 和迭代时间 benchmark 不属于本报告范围。本报告中的字节数是 optimizer 持久状态张量的精确计数，不是 CUDA allocator peak memory 或进程 RSS。

### 1.2 结论

实现和实验得到以下结论：

1. staff 提供的 `ToyModel` 和 tied-weight 模型测试连续运行 5 次，共 10 个参数化 case，全部通过。
2. 新增测试覆盖了多参数组、训练中动态 `add_param_group()`、学习率调度器、完整梯度清理和 optimizer state 唯一归属。
3. 在 2-rank CPU/Gloo 实验中，普通 AdamW 每 rank 保存 `131,088 B` 状态；分片后每 rank 保存 `65,544 B`，恰好下降 `50%`。
4. 10 个训练 step 中，相对非分片 AdamW 的最大参数误差为 `0.0`，rank 间最大参数误差也是 `0.0`。
5. 模型参数和梯度仍然是完整复制的；本实现只消除 optimizer state 的跨 rank 冗余。

## 2. 背景知识与理论

### 2.1 DDP 为什么复制 optimizer state

设数据并行 world size 为 $N$。普通 DDP 的每个 rank 都持有完整参数 $\theta$，在 backward 后通过 all-reduce 得到相同的平均梯度 $g_t$。若所有 rank 从相同参数和 optimizer state 开始，并执行相同的确定性更新，则它们会得到相同的新参数。

这个方案简单，但每个 rank 都保存同一份 optimizer state。增加 rank 数不会降低单 rank 的 optimizer state 内存。

### 2.2 AdamW 为什么需要两份参数规模的状态

对每个参数元素，AdamW 维护一阶矩 $m_t$ 和二阶矩 $v_t$：

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2. $$

忽略 bias correction 和 weight decay 的展开，更新可写成：

$$ \theta_{t+1} = (1-\eta\lambda)\theta_t - \eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}. $$

若 FP32 参数占 $M$ 字节，则两个 FP32 moment 约占 $2M$ 字节。PyTorch AdamW 还为每个已经初始化状态的参数保存一个标量 `step` 张量，因此实测值会比 $2M$ 略大。

PyTorch optimizer state 通常是惰性创建的：构造 optimizer 时还没有 `exp_avg` 和 `exp_avg_sq`；第一次对具有梯度的参数执行 `step()` 后才创建。因此比较状态内存必须放在至少一次 optimizer step 之后。

### 2.3 只分片 optimizer state 后的内存模型

以 FP32 训练为例，暂不计 activation、临时 buffer 和 allocator 碎片。普通 DDP 每 rank 的主要训练状态近似为：

$$ M_{\mathrm{DDP}} \approx M_{\theta}+M_g+M_m+M_v = 4M. $$

只对 AdamW optimizer state 分片后，理想均衡情况下每 rank 近似为：

$$ M_{\mathrm{sharded}} \approx M_{\theta}+M_g+\frac{M_m+M_v}{N} = \left(2+\frac{2}{N}\right)M. $$

当 $N=2$ 时，总量从约 $4M$ 降为约 $3M$，即完整训练状态约减少 25%；但 optimizer state 自身从 $2M$ 降为 $M$，减少 50%。

这里不能把“optimizer state 减半”写成“总显存减半”，因为参数和梯度仍然完整复制。若使用 FP32 master weight、混合精度副本或 gradient scaler，还要把这些额外状态单独加入模型。

### 2.4 Owner-update-broadcast 协议

对第 $i$ 个参数指定唯一 owner $o(i)\in\{0,\ldots,N-1\}$。optimizer state sharding 不能省略梯度归约：owner 必须使用所有 local-batch gradient 的归约结果，而不是只使用自己的局部梯度。

本仓库当前的 `DDP + ShardedOptimizer` 组合采用教学版的简单路径：

1. 每个 rank $r$ 通过 backward 得到局部梯度 $\widetilde g_{t,r}^{(i)}$；
2. 现有 DDP 对局部梯度执行 all-reduce average，使每个 rank 都持有全部参数的完整全局梯度 $g_t^{(i)}$；
3. 只有 rank $o(i)$ 使用同步后的 $g_t^{(i)}$ 和本地状态 $s_t^{(i)}$ 更新参数 $\theta_t^{(i)}$；
4. 所有 rank 共同执行以 $o(i)$ 为 source 的 parameter broadcast。

梯度同步可写为：

$$ g_t^{(i)}=\frac{1}{N}\sum_{r=0}^{N-1}\widetilde g_{t,r}^{(i)}. $$

从 owner 更新的数据依赖看，每个 rank 确实只需要自己负责参数的归约梯度，不需要拿到其他参数的完整梯度。更高效的实现可以执行 reduce-scatter：梯度在归约的同时按 owner 分发，每个 rank 只收到自己的 gradient shard；owner 更新后，再 all-gather 更新后的 parameter shard。标准 ring all-reduce 本来就可分解为 reduce-scatter 加 all-gather，因此可以用“gradient reduce-scatter + updated-parameter all-gather”替代“完整 gradient all-reduce + parameter broadcast”。

当前 `ShardedOptimizer` 没有实现这条优化路径：它完全不参与 gradient communication，只消费调用方已经放入 `parameter.grad` 的梯度，因此只近似实现 optimizer-state partitioning（$P_{os}$，通常称为 Stage 1）的内存语义。若进一步在 backward 期间只保留 owner 的 gradient shard 并释放其他梯度存储，就叠加了 gradient partitioning 阶段 $P_g$，整体成为 $P_{os+g}$（通常称为 Stage 2）。

更新关系为：

$$ \theta_{t+1}^{(i)} = U\!\left(\theta_t^{(i)},g_t^{(i)},s_t^{(i)}\right),\qquad \theta_{t+1,r}^{(i)}\leftarrow\operatorname{broadcast}_{o(i)}\!\left(\theta_{t+1}^{(i)}\right). $$

其中 $U$ 是底层 optimizer 对该参数的更新函数。

### 2.5 正确性为什么成立

假设：

1. step 开始前所有 rank 的参数一致；
2. owner 使用的梯度等于普通非分片 optimizer 使用的梯度；
3. 参数更新只依赖该参数、该参数的状态和所属参数组超参数；
4. 所有 rank 使用相同 owner 映射和 collective 顺序。

owner 执行的更新就与普通 optimizer 对该参数执行的更新相同。broadcast 完成后，所有 rank 又持有 owner 的新参数，因此所有参数重新一致。对训练 step 做归纳即可得到整段训练与普通 optimizer 等价。

关键前提是“梯度已经正确同步”。`ShardedOptimizer` 不负责 DDP gradient all-reduce；它应放在 DDP 的 `finish_gradient_synchronization()` 之后执行。

### 2.6 与 ZeRO 的关系

ZeRO 将数据并行冗余分为三个逐步扩展的阶段：

| 阶段 | ZeRO 原文记号 | 分片对象 |
|---|---|---|
| ZeRO Stage 1 | $P_{os}$ | optimizer state |
| ZeRO Stage 2 | $P_{os+g}$ | optimizer state 和 gradient |
| ZeRO Stage 3 | $P_{os+g+p}$ | optimizer state、gradient 和 parameter |

本实验在“只让一个 rank 保存并更新每个参数的 optimizer state”这一点上接近 ZeRO Stage 1，但只是教学版协议：

1. 本实现不接管 gradient synchronization，完整 gradient 仍可在所有 rank 上存在。
2. 本实现逐参数 broadcast 更新后的参数，产生 $K$ 次 collective 固定开销，其中 $K$ 是唯一参数张量数。
3. 工业实现通常按 bucket 执行 reduce-scatter 和 all-gather，减少 collective 次数并改善通信调度。
4. 本实现没有 checkpoint reshard、elastic world size、通信重叠或拓扑感知放置。

## 3. 实现设计

### 3.1 文件与职责

| 文件 | 职责 |
|---|---|
| `cs336_systems/sharded_optimizer.py` | `ShardedOptimizer` 核心实现 |
| `tests/adapters.py` | 将作业 adapter 接到实现 |
| `tests/test_sharded_optimizer_variants.py` | 参数组、scheduler 和状态分片回归测试 |
| `scripts/experiment_sharded_optimizer.py` | 2-rank CPU/Gloo 可复现实验 |
| `notes/assets/sharded_optimizer/cpu_gloo_correctness.json` | 固化的实验配置、环境和结果 |

### 3.2 为什么同时保留“完整外层组”和“本地内层组”

wrapper 有两个不同职责：

1. 外层 `ShardedOptimizer.param_groups` 保存完整参数集。
2. 内层 `local_optimizer.param_groups` 只保存当前 rank 拥有的参数。

完整外层组是必要的，因为调用 `sharded_optimizer.zero_grad()` 时必须清理所有复制参数的 gradient，而不只是本 rank 拥有的参数。内层组则保证 AdamW 只为 owner 参数创建 `exp_avg`、`exp_avg_sq` 和 `step`。

如果 wrapper 只暴露本地参数组，非 owner 参数的 `.grad` 会跨 step 残留和累加；如果内层 optimizer 接收完整参数组，又无法节省状态。

### 3.3 父类构造与 `add_param_group()` 生命周期

PyTorch `Optimizer.__init__()` 会在构造期间调用多态的 `self.add_param_group()`。因此不能假设 subclass 的 `add_param_group()` 只在构造完成后运行。

实现先设置 owner 映射、rank load 和 `_initializing`，再调用父类构造。构造期间只登记完整参数组和 owner；父类构造结束后，再一次性创建本地 optimizer。训练期间调用 `add_param_group()` 时，则同时向外层和本地 optimizer 添加对应组。

这同时支持逐步解冻模型：

```python
optimizer.add_param_group(
    {
        "params": newly_unfrozen_parameters,
        "lr": 1e-4,
    }
)
```

前提是所有 rank 以相同参数顺序执行相同调用。

### 3.4 参数 owner 如何分配

对每个首次出现的参数，选择当前累计参数字节最少的 rank；若 load 相同，选择较小 rank。参数权重为：

$$ w_i = \operatorname{numel}(\theta_i)\cdot\operatorname{element\_size}(\theta_i). $$

这个确定性 greedy list scheduling 比按“参数张量个数”轮转更接近 optimizer state 的实际字节平衡，尤其能避免一个 rank 恰好拿到多个大矩阵、另一个 rank 只拿到 bias。

它仍然是参数粒度分片：单个大张量不能再拆分，所以最大参数可能主导不均衡。FSDP/ZeRO 的 flat partition 可以进一步切分张量。

### 3.5 固定 collective 顺序

每个首次出现的参数都会被加入 `_parameters_in_broadcast_order`。`step()` 严格按该顺序执行 broadcast。只要每个 rank 使用相同参数组和参数顺序，collective 序列就是：

```text
broadcast(parameter_0, src=owner_0)
broadcast(parameter_1, src=owner_1)
...
broadcast(parameter_K-1, src=owner_K-1)
```

collective 顺序不一致会导致错误匹配或死锁，因此 owner 映射不仅是内存分配信息，也是分布式协议的一部分。

对 tied weights，标准 `module.parameters()` 默认按 parameter identity 去重；实现中的 owner map 也只为首次出现的 parameter identity 建立一个 broadcast 项。

### 3.6 参数组超参数与 scheduler

学习率调度器修改的是外层 `sharded_optimizer.param_groups`。本地 optimizer 持有独立的参数子集 group，如果不额外同步，scheduler 修改的 `lr` 不会进入真实更新。

因此每次本地 `step()` 前，wrapper 将外层 group 中除 `params` 和 `param_names` 以外的选项同步到对应本地 group。测试使用 `ExponentialLR(gamma=0.5)` 验证分片 optimizer 与 baseline 在三步后仍严格一致。

### 3.7 Rank-local checkpoint

`state_dict()` 返回当前 rank 的本地 optimizer shard，`load_state_dict()` 也加载当前 rank 的 shard。这样不会在保存 checkpoint 时重新复制完整 optimizer state。

其直接约束是：

1. 每个 rank 要保存自己的 optimizer state 文件；
2. 恢复时需要相同 world size、参数顺序和 owner 规则；
3. 当前实现不支持从 $N$ 个 shard 自动 reshard 到另一个 world size。

## 4. 关键代码

### 4.1 构造和本地 optimizer

构造函数先读取默认进程组的 rank/world size，再让父类登记完整参数组，最后按 owner 过滤出本地组：

- [`ShardedOptimizer.__init__`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L26-L64)
- [`_build_local_optimizer`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L218-L229)
- [`_local_param_group`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L179-L197)

### 4.2 Owner 分配

[`_assign_owner_ranks`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L148-L155) 维护每个 rank 已分配的参数字节，以 `(load, rank)` 作为稳定比较键。

### 4.3 本地更新和参数同步

[`step`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L107-L125) 先把最新参数组超参数同步给本地 optimizer，转发 `closure` 和 optimizer-specific keyword arguments，再按稳定顺序广播全部参数。

`parameter.detach()` 只切断 autograd 视图，不复制 parameter storage；`dist.broadcast()` 原地写入非 owner rank 的同一参数 storage。

### 4.4 动态参数组

[`add_param_group`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L80-L105) 先复用 PyTorch 父类的参数检查和默认值处理，再为新参数分配 owner。训练期调用还会执行 [`_add_local_param_group`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L157-L177)。

### 4.5 Scheduler 和 checkpoint group 同步

- [`_sync_local_group_options`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L233-L244)：step/save 前从外层同步到内层。
- [`_adopt_loaded_group_options`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/sharded_optimizer.py#L246-L257)：load 后从 checkpoint 的内层 group 恢复外层选项。

### 4.6 Adapter

作业入口 [`get_sharded_optimizer`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/adapters.py#L121-L137) 直接构造 `ShardedOptimizer`，不在 adapter 中复制实现逻辑。

## 5. 实验方法

### 5.1 环境

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon Platinum 8336C @ 2.30 GHz |
| 可用 CPU | 56 cores |
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu130 |
| Backend | Gloo |
| World size | 2 |
| 每 worker CPU threads | 1 |
| CUDA available | `False` |

限制每个 worker 使用一个 PyTorch/BLAS thread，避免两个本地进程各自占满 56 个 CPU core。

### 5.2 Staff correctness 测试

handout 建议将测试运行 5 次。本实验执行：

```bash
for run in 1 2 3 4 5; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uv run pytest tests/test_sharded_optimizer.py -q
done
```

每次包含：

1. `ToyModel`；
2. `ToyModelWithTiedWeights`；
3. 普通 AdamW 与分片 AdamW 连续训练 10 step；
4. 最终逐参数比较。

### 5.3 扩展接口测试

执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run pytest \
    tests/test_sharded_optimizer.py \
    tests/test_sharded_optimizer_variants.py -q
```

扩展测试 [`test_sharded_optimizer_supports_parameter_groups_and_partitions_state`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/test_sharded_optimizer_variants.py#L13-L80) 检查：

1. 两个不同学习率的 parameter group；
2. 构造后动态 `add_param_group()`；
3. `ExponentialLR` 对真实本地更新生效；
4. 所有 rank 得到相同 owner 序列；
5. 每个 AdamW state 只存在于 owner rank；
6. 跨 rank 汇总后的 `exp_avg` 元素数恰好等于完整参数元素数；
7. 外层 `zero_grad()` 清理全部复制参数。

### 5.4 状态分片实验

执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python scripts/experiment_sharded_optimizer.py \
    --world-size 2 \
    --steps 10 \
    --width 64 \
    --batch-size 8 \
    --num-threads 1 \
    --output benchmark_results/sharded_optimizer/cpu_gloo_correctness.json
```

实验模型由 4 个无 bias 的 $64\times64$ 线性层组成，共 4 个等大的 FP32 参数张量：

$$ 4\cdot64\cdot64 = 16{,}384\text{ elements}=65{,}536\text{ B}. $$

每个 rank 使用相同 seed、输入和 target，使 gradient 相同。这个设置有意隔离 optimizer state sharding，不把 DDP gradient all-reduce 的误差或耗时混入结果。

每一步同时训练：

1. 完整非分片 AdamW baseline；
2. `ShardedOptimizer(AdamW)`。

### 5.4.1 两个独立的正确性指标

训练循环内的检查比较同一 rank 上的 baseline 参数和 sharded 参数。每次 optimizer step 后，对每个参数计算最大逐元素绝对误差，并用 `maximum_baseline_error` 保留所有 step、所有参数中的最大值。这个指标回答：

> 分片 optimizer 是否执行了与完整 AdamW 相同的参数更新？

这项检查没有发起 collective，因此可以在每个 step 后执行，并能保留训练中途出现过但最终可能消失的误差。

训练循环外的检查比较不同 rank 的 sharded 参数。每个 rank 先执行：

```python
rank_zero_parameter = parameter.detach().clone()
dist.broadcast(rank_zero_parameter, src=0)
```

rank 0 的副本保留 rank 0 参数，其他 rank 的副本被 broadcast 覆盖为 rank 0 参数；随后每个 rank 比较自己的真实 `parameter` 与这个副本。使用 `detach().clone()` 有两个目的：

1. 校验不进入 autograd graph；
2. broadcast 只修改临时副本，不会把错误的本地模型参数“修复”成 rank 0 参数。

这个指标回答：

> optimizer step 后所有模型副本是否仍然一致？

两个指标不能互相替代。所有 rank 可能同步到了相同但错误的值，此时跨 rank 误差为零，但 baseline 误差不为零；也可能 rank 0 更新正确、其他 rank 未收到更新，此时 rank 0 的 baseline 误差为零，但其他 rank 的跨 rank 误差不为零。

跨 rank 检查放在训练循环外，只验证最终同步不变量，避免每一步额外增加一轮仅用于诊断的 parameter broadcast。训练热路径中真正的参数同步仍由 `ShardedOptimizer.step()` 完成。

### 5.4.2 本地状态统计与跨 rank 汇总

每个 rank 分别统计自己的 optimizer shard：

| 指标 | 含义 |
|---|---|
| `optimizer_state_entries` | 本地 optimizer 中已经创建 state 的参数数量 |
| `optimizer_state_tensor_bytes` | `step`、`exp_avg`、`exp_avg_sq` 等状态张量的总字节数 |
| `owned_parameter_count` | owner 为当前 rank 的参数张量数量 |
| `owned_parameter_elements` | 本地 owner 参数的元素总数 |
| `owned_parameter_bytes` | 本地 owner 参数的存储字节数 |

`local_optimizer is None` 是合法边界：当 world size 大于参数数量时，某些 rank 可能没有分到任何参数。这些 rank 的 state entries 和 state tensor bytes 都记为零，但仍必须参加后续 collective。

`local_owned_parameters` 通过稳定参数顺序与 `parameter_owners` 逐项配对，再筛选 `owner == rank`。因此“参数数量”“元素数量”和“字节数”是三个不同口径：参数张量数量相同不代表 state 内存相同，负载均衡主要应观察字节数。

每个 rank 将以上指标和两个正确性误差组成 `local_result`，然后执行：

```python
rank_results = [None] * world_size
dist.all_gather_object(rank_results, local_result)
```

`all_gather_object` 将 Python 字典序列化并收集，使每个 rank 最终都得到按 rank 编号排列的完整列表：

```text
[rank_0_result, rank_1_result, ..., rank_N-1_result]
```

它是 collective，因此所有 rank 都必须以相同顺序调用。它适合这里的小规模实验 metadata，不适合训练热路径中的大 tensor 通信，因为 Python object serialization 比直接 tensor collective 更慢，也无法代表训练通信性能。

虽然每个 rank 都获得完整结果，只有 rank 0 负责写 JSON。rank 0 会先确认收到了全部 $N$ 份结果，再从所有 rank 的记录中取全局最大 baseline error 和跨 rank error；最终 `status` 基于这两个全局值判断，不能只检查 rank 0 的局部误差。

脚本 [`_worker`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/experiment_sharded_optimizer.py#L66-L210) 完成训练、两类正确性检查、本地状态统计和跨 rank 结果汇总。

固化结果见 [`cpu_gloo_correctness.json`](./assets/sharded_optimizer/cpu_gloo_correctness.json)。

## 6. 实验结果

### 6.1 测试结果

| 测试 | 运行次数 | 结果 |
|---|---:|---:|
| staff `ToyModel` | 5 | 5 passed |
| staff tied-weight model | 5 | 5 passed |
| 扩展参数组/state/scheduler 测试 | 1 | passed |

完整联合命令结果为 `3 passed`。5 次 staff 重复均为 `2 passed`。

### 6.2 Optimizer state 字节数

| 指标 | 普通 AdamW | Sharded rank 0 | Sharded rank 1 |
|---|---:|---:|---:|
| 完整模型参数 bytes/rank | 65,536 | 65,536 | 65,536 |
| owner 参数 bytes | 65,536 | 32,768 | 32,768 |
| optimizer state entries | 4 | 2 | 2 |
| optimizer state tensor bytes | 131,088 | 65,544 | 65,544 |
| 相对普通 optimizer state | 100% | 50% | 50% |

`131,088 B` 的组成是：

$$ 2\cdot65{,}536\text{ B}+4\cdot4\text{ B}=131{,}088\text{ B}. $$

前一项是 `exp_avg` 和 `exp_avg_sq`，后一项是 4 个 FP32 `step` 标量。每个 shard 拥有两个参数，因此为：

$$ 2\cdot32{,}768\text{ B}+2\cdot4\text{ B}=65{,}544\text{ B}. $$

两个 shard 的状态总和仍是 `131,088 B`，恰好等于一份完整 AdamW 状态；普通 2-rank DDP 则会跨集群保存 `262,176 B`。这说明状态没有丢失，只是从“每 rank 一份”变成“全体 rank 合计一份”。

### 6.3 数值正确性

| 指标 | rank 0 | rank 1 |
|---|---:|---:|
| 相对完整 AdamW 的最大参数误差 | 0.0 | 0.0 |
| 相对 rank 0 的最大参数误差 | 0.0 | 0.0 |

owner 序列为 `[0, 1, 0, 1]`。4 个参数大小相同，因此两个 rank 各拥有两个参数，达到理想均衡。

### 6.4 通信口径

当前实验每 step 发起 4 次 parameter broadcast，参数 payload 之和为 `65,536 B`。这里的 payload 是所有被广播参数的逻辑张量字节之和，不等于底层网络链路累计流量；实际链路流量取决于 broadcast 算法和拓扑。

本实验没有报告有统计意义的 runtime benchmark。CPU/Gloo correctness 时间不能回答 handout 下一题要求的双 GPU/NCCL `xl` 训练速度。

## 7. 复杂度与可并行性

设唯一参数张量数为 $K$，参数总字节为 $M$，world size 为 $N$，rank $r$ 拥有参数字节为 $M_r$。

### 7.1 复杂度

| 阶段 | 时间或通信复杂度 | 额外持久状态 |
|---|---|---|
| owner 分配 | $O(KN)$ | $O(K+N)$ Python metadata |
| 本地 optimizer step | 与 $M_r$ 成正比 | 约 $2M_r$ AdamW moments |
| 参数同步 | $K$ 次 broadcast，逻辑 payload 合计 $M$ | 无另一份完整参数持久副本 |
| `zero_grad()` | 扫描完整参数集 | 不新增持久状态 |

owner 分配中的 $O(KN)$ 来自每个参数扫描 rank load。训练中的 $N$ 通常远小于参数张量数，而且分配只发生在构造或动态加组时，因此没有引入 heap 结构。

### 7.2 可并行部分

各 rank 的本地 optimizer update 作用于不相交参数集合，可以并行执行。当前实现等本地 update 完成后，所有 rank 以同一顺序同步执行 broadcast。

可继续优化的方向：

1. 把多个小参数打包成 bucket，降低 $\alpha$ latency；
2. 使用异步 collective，把已经更新的 bucket broadcast 与剩余 optimizer update 重叠；
3. 按 flat state partition 切分超大张量，改善负载均衡；
4. 将 gradient reduce-scatter 与 parameter all-gather 组织成 ZeRO Stage 1 风格协议。

这些优化会扩大协议状态机和测试范围，不属于当前 correctness 实现。

## 8. 使用方式

与本仓库 overlapped DDP 组合时，训练顺序为：

```python
ddp_model = OverlappedDistributedDataParallel(model)
optimizer = ShardedOptimizer(
    ddp_model.parameters(),
    torch.optim.AdamW,
    lr=1e-3,
    weight_decay=0.01,
)

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(ddp_model(inputs), targets)
loss.backward()
ddp_model.finish_gradient_synchronization()
optimizer.step()
```

数学上参数和梯度可视为列向量；PyTorch 实现中 parameter tensor 保持模型原始 shape，optimizer 逐元素更新，broadcast 不改变 tensor layout。

## 9. 实现边界

1. 必须先初始化默认 distributed process group。
2. 所有 rank 必须以相同顺序构造参数组、动态添加参数组并调用 `step()`。
3. optimizer step 前必须由 DDP 或其他机制确保 owner 看到正确的全局 gradient。
4. 本实现只适合更新可按参数分解的 optimizer，例如 SGD、Adam 和 AdamW；具有跨参数全局状态或不同 rank 可能执行不同 closure 次数的 optimizer 需要额外协议。
5. 若 closure 内含 collective，所有 rank 必须执行一致的 collective 序列。
6. checkpoint 是 rank-local shard，不支持直接改变 world size 后恢复。
7. 参数、gradient 和 activation 仍未分片。
8. 当前逐参数同步会放大小张量的 collective latency。
9. 当前没有主动跨 rank 校验参数 metadata；不一致可能表现为 collective error 或 deadlock。
10. 本报告的状态张量计数不是 allocator peak-memory 测量。

## 10. 参考资料

1. Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
2. PyTorch, [`torch.optim.Optimizer`](https://docs.pytorch.org/docs/stable/optim.html).
3. PyTorch, [`ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/stable/distributed.optim.html).
