# Fully Sharded Data Parallel 实验报告

## 1. 问题与结论

### 1.1 问题

前一节的 optimizer-state sharding 只消除了 AdamW state 的重复副本。每个 rank 仍保留完整参数和完整梯度，因此模型参数本身超过单卡容量时仍然无法训练。

本实验实现教学版 `FullyShardedDataParallel`（FSDP）：

1. `Linear` 和 `Embedding` 的 FP32 master weight 按 rank 分片常驻；
2. forward/backward 使用前 all-gather 完整 weight；
3. 使用后立即恢复为 master shard；
4. 完整 weight gradient 通过 reduce-scatter 变为本地 gradient shard；
5. RMSNorm 等小参数保持复制，并使用 all-reduce 同步梯度；
6. mixed precision 时先把 master shard 转为 compute dtype，再通信和计算。

### 1.2 结论

1. staff 的 FP32/FP16 correctness 与 gradient-sync 共 4 个 case 全部通过；额外 padding/state 和非注册顺序执行测试也通过。
2. FP32 master weight、gradient 和 AdamW state 都保持 shard shape，标准 optimizer 可以直接更新。
3. 非整除参数会先补零再均分；all-gather 后裁掉 padding 并恢复原 shape。
4. 首轮训练记录真实的 forward/backward 模块执行顺序；后续迭代在层 $i-2$ 完成后预取层 $i$，layer pre-hook 只等待未完成的尾部。
5. FSDP 节省的是长期驻留的模型状态；完整 weight 和完整 gradient 仍会按层短暂出现，activation 默认仍由 autograd 保存。

## 2. 背景知识

### 2.1 DDP、ZeRO 与 FSDP 的数据驻留

设模型 FP32 参数总大小为 $M$，暂不计 activation 和通信临时 buffer。AdamW 有两份参数规模的 moment。

| 策略 | parameter/rank | gradient/rank | AdamW moments/rank |
|---|---:|---:|---:|
| 普通 DDP | $M$ | $M$ | $2M$ |
| optimizer-state sharding | $M$ | $M$ | 约 $2M/N$ |
| FSDP | 约 $M/N$ | 约 $M/N$ | 约 $2M/N$ |

FSDP 的核心不是“少算参数”，而是改变参数、梯度和 optimizer state 的常驻布局。计算某层时仍需要完整 weight，只是完整 weight 作为短生命周期 materialized buffer 存在。

### 2.2 Master weight、master shard 与 local gradient shard

Master weight 是 optimizer 更新所依据的高精度权威参数，mixed precision 训练中通常使用 FP32。它是一个逻辑上的完整参数；FSDP 不要求任一 rank 长期保存它的完整物理副本。

Master shard 是当前 rank 长期保存并交给 optimizer 的那一段 FP32 master weight。例如将 $w=[w_0,\ldots,w_7]$ 分布到两个 rank：

```text
rank 0 master shard: [w0, w1, w2, w3]
rank 1 master shard: [w4, w5, w6, w7]
```

Local gradient shard 是 reduce-scatter 输出中与本地 master shard 对应的部分。这里的“local”描述梯度的存储归属，不表示它只来自 local batch：它已经聚合所有 rank 的局部梯度。例如 rank 0 得到全局平均梯度的前半段，rank 1 得到后半段；两个 rank 随后分别更新自己的 master shard。

### 2.3 数学布局与 PyTorch 布局

数学上以列向量记线性层：

$$ y=Wx,\qquad W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},\quad x\in\mathbb{R}^{d_{\mathrm{in}}}. $$

PyTorch 特征位于最后一维，对应实现为 $y=xW^\top$，weight tensor shape 仍为 `(d_out, d_in)`。FSDP 将 weight flatten 后按存储元素切片，不改变完整 weight 被 materialize 后的数学含义。

### 2.4 Weight all-gather

设第 $l$ 层完整 weight flatten 后为 $w_l$，world size 为 $N$。每个 rank 常驻 shard $w_l^{(r)}$：

$$ w_l=\operatorname{all\text{-}gather}\left(\{w_l^{(r)}\}_{r=0}^{N-1}\right). $$

all-gather 后，将拼接结果裁掉 padding 并 reshape 为原 weight shape。forward 完成后，参数对象的 `.data` 恢复指向 FP32 master shard。

完整 compute weight 可以在 forward 后释放，是因为 optimizer 尚未更新参数，各 rank 的 master shard 仍可在 backward 前无损重建同一个 weight。

### 2.5 Gradient reduce-scatter

每个 rank 根据自己的 local batch 得到完整参数的局部梯度 $\widetilde g_l^{(r)}$。只需要保留与本地 parameter shard 对应的全局平均梯度：

$$ g_l^{(r)}=\frac{1}{N}\operatorname{reduce\text{-}scatter}\left(\{\widetilde g_l^{(j)}\}_{j=0}^{N-1}\right)_r. $$

reduce-scatter 同时完成跨 rank 求和和结果切分。完成后，`parameter.data` 与 `parameter.grad` 都具有相同 shard shape，标准 AdamW/SGD 无需理解 FSDP。

在 parameter post-accumulate hook 开始执行时，当前 rank 会短暂同时持有当前层的完整 weight、完整 gradient、padded gradient 通信输入、本地 master shard 和 reduce-scatter 输出 shard。hook 启动 reduce-scatter 后立即清除完整 gradient 并恢复 master shard，所以这是逐层的瞬时开销，不是整个模型的完整 gradient 常驻。

### 2.6 为什么 RMSNorm 不分片

RMSNorm weight 只有 $d_{\mathrm{model}}$ 个元素，而 Linear weight 通常有 $d_{\mathrm{in}}d_{\mathrm{out}}$ 个元素。对很小的 norm 参数执行 all-gather/reduce-scatter，固定 collective latency 可能远大于节省的内存和带宽。

这里的 `Linear` 和 `Embedding` 是递归匹配的模块类型，不是只指模型顶层的两个参数。因此 Transformer 中以下 weight 都会分片：

- `token_embeddings.weight`：Assignment 1 `Embedding`；
- 每层 attention 的 `q_proj.weight`、`k_proj.weight`、`v_proj.weight` 和 `output_proj.weight`：Assignment 1 `Linear`；
- 每层 FFN 的 `w1.weight`、`w2.weight` 和 `w3.weight`：Assignment 1 `Linear`；
- `lm_head.weight`：Assignment 1 `Linear`；
- 测试或其他模型中的 PyTorch `nn.Linear.weight` 和 `nn.Embedding.weight`。

因此，一个 32 层 Transformer 共有 $32\times(4+3)+2=226$ 个 shardable weight tensor。保持完整复制的是每层两个 RMSNorm weight 和最终 RMSNorm weight，共 $32\times2+1=65$ 个；RoPE 只有 buffer，没有可训练参数。若使用带 bias 的 PyTorch `nn.Linear`，当前实现只分片其 `weight`，`bias` 也保持复制。所有复制参数的梯度通过异步 all-reduce 求平均。

### 2.7 Mixed precision

反复累积更新的 master weight 和 optimizer moments 对精度敏感，因此保持 FP32。设 compute dtype 为 FP16：

```text
FP32 master shard
  -> cast local shard to FP16
  -> all-gather FP16 full weight
  -> FP16 forward/backward
  -> full gradient cast to FP32
  -> reduce-scatter FP32 gradient shard
  -> FP32 optimizer update
```

这样 weight all-gather payload 减半，同时 optimizer 更新仍使用 FP32。Gradient reduce-scatter 在当前实现中使用 FP32，以保证 master-gradient dtype 与 optimizer 参数一致。

### 2.8 Weight 与 activation 的生命周期

以列向量线性层 $y=Wx$ 为例，backward 中 $\nabla_xL=W^\top\nabla_yL$ 需要 weight，而 $\nabla_WL=(\nabla_yL)x^\top$ 需要 forward activation $x$。两者虽然都参与 backward，但能否释放不同：

| 数据 | forward 后能否释放 | backward 时如何获得 |
|---|---|---|
| 完整 weight | 可以 | 从未更新的 FP32 master shards 再次 all-gather |
| 普通 activation | 默认不能 | 由 autograd 保存到对应 backward |
| checkpoint 区域内的中间 activation | 可以 | 保存区域边界，backward 时重新执行该区域的 forward |

因此训练主链路的生命周期是：

```text
forward:
  all-gather 当前层完整 weight
  -> 计算并保存 backward 所需 activation
  -> reshard weight

backward:
  再次 all-gather 当前层完整 weight
  -> 使用 weight、activation 和 grad_output 计算梯度
  -> reduce-scatter 完整 weight gradient
  -> 释放完整 weight 和完整 gradient
```

FSDP 解决 parameter、gradient 和 optimizer state 的常驻冗余，不会自动消除 activation。Activation checkpointing 是独立的时间换空间策略，通常以 Transformer block 为粒度，只保存 block 边界并在 backward 中重算内部 activation；它可能额外触发 weight all-gather。当前教学实现要求每个 shardable module 每轮 forward/backward 各执行一次，因此暂不支持 checkpoint 重计算，后者需要扩展执行顺序和 collective 调度状态机。

### 2.9 Prefetch 为什么是两层 lookahead

若等到层 $i$ 的 pre-hook 才启动 all-gather，计算必须完整等待通信。handout 要求层 $i-2$ forward 完成后开始 gather 层 $i$：

```text
layer i-2 compute
  -> launch all-gather(i)
layer i-1 compute overlaps all-gather(i)
  -> pre-hook(i) waits only for remaining tail
layer i compute
```

设层 $i$ 的 all-gather 时间为 $C_i$，中间层 $i-1$ 的计算时间为 $F_{i-1}$。忽略调度和资源竞争时，使用前暴露等待近似为：

$$ W_i=\max(0,C_i-F_{i-1}). $$

`async_op=True` 只创造 overlap 机会；只有 `W_i` 接近零，或 GPU timeline 显示 NCCL kernel 在使用前结束，才能说 prefetch 及时。

模块注册顺序不一定等于执行顺序。例如 SwiGLU 通常按 `w1 -> w3 -> w2` 执行，但 `named_modules()` 的注册顺序可能是 `w1 -> w2 -> w3`。因此实现会在首轮训练中按需 materialize 并分别记录真实 forward/backward 顺序，从第二轮开始再按已验证的顺序预取；这避免了错误预取已经完成反向传播的 weight。

## 3. 实现设计

### 3.1 文件职责

| 文件 | 职责 |
|---|---|
| `cs336_systems/fsdp.py` | 分片状态、hook、all-gather、reduce-scatter 和参数重建 |
| `cs336_systems/fsdp_observer.py` | 定义可选的 collective lifecycle observer seam |
| `tests/adapters.py` | staff 测试入口 |
| `tests/test_fsdp_variants.py` | padding、state locality 与更新后参数一致性 |

### 3.2 初始化与分片

[`_ShardedWeight.__init__`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L36-L65) 保存：

- 原始 shape 和元素数；
- 每 rank 固定 shard 元素数；
- padding 后总元素数；
- FP32 master shard；
- pending all-gather 和 materialized 状态。

若完整参数元素数为 $P$，则：

$$ P_{\mathrm{shard}}=\left\lceil\frac{P}{N}\right\rceil,\qquad P_{\mathrm{padded}}=NP_{\mathrm{shard}}. $$

初始化先从 rank 0 广播所有参数和 buffer，再切 shard，避免不同 rank 随机初始化不一致。

### 3.3 Forward 生命周期

[`forward`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L162-L172) 在执行顺序已知时先启动前两层 all-gather；首轮 forward 用于发现真实执行顺序。

每个 shardable layer：

1. forward pre-hook 调用 [`materialize`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L86-L98)，等待必要的 tail 并安装完整 weight；
2. layer 执行真实计算；
3. forward post-hook 在输出 Tensor 上注册 backward 触发器；
4. 恢复 FP32 master shard；
5. 按真实执行顺序启动当前位置之后第二层的 all-gather。

完整 compute weight 不进入 optimizer 参数集合；optimizer 始终看到同一个 Parameter 对象，只是该对象在计算窗口中临时切换 `.data` storage。

### 3.4 Backward 生命周期

forward output 上的 Tensor hook 在对应模块 backward 开始前触发：

1. 记录当前层的真实 backward 位置；
2. materialize 当前层完整 compute weight；
3. 在执行顺序已知时启动后续一层和两层的预取；
4. 返回原 output gradient，不改变 autograd 数值。

weight gradient 累积完成后，parameter post-accumulate hook：

1. flatten 并补齐完整 gradient；
2. 启动异步 reduce-scatter；
3. 清除完整 `parameter.grad`；
4. 恢复 FP32 master shard；
5. 保存 `Work` 和输出 shard。

[`finish_gradient_synchronization`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L174-L189) 验证 backward 执行顺序，等待所有 reduce-scatter/all-reduce，除以 world size，再把 gradient shard 安装到 Parameter 上。

### 3.5 完整参数重建

[`gather_full_parameters`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L191-L198) 对 master shard 执行 FP32 all-gather，裁 padding 后恢复原 shape；复制参数直接 clone。

这个接口用于 correctness 检查和模型导出，不改变 optimizer 当前持有的 shard。

### 3.6 观测逻辑与训练主链路解耦

[`FSDPObserver`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_observer.py#L13-L29) 是可选的观测 seam。FSDP 只发送 all-gather 的 `launched`、`waiting`、`finished` 生命周期事件；默认不创建 recorder，也不保存时间序列。计时、ready 检查、通信字节核算与 NVTX wait 标记由 accounting adapter 实现，因此不会污染核心分片状态和训练接口。

### 3.7 Collective 顺序

所有 rank 必须：

1. 首轮训练以相同顺序执行所有 shardable module；
2. 后续迭代保持相同的 forward/backward 执行顺序；
3. 以相同顺序触发 all-gather 和 reduce-scatter；
4. 在下一次 forward 前调用 `finish_gradient_synchronization()`。

动态控制流、rank-dependent unused layer 或不同模块调用顺序会破坏 collective 匹配，可能产生 hang。

## 4. 关键代码

| 功能 | 代码 |
|---|---|
| shard/padding | [`_ShardedWeight.__init__`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L36-L65) |
| async all-gather | [`start_all_gather`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L71-L84) |
| materialize/reshape | [`materialize`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L86-L98) |
| forward/backward hooks | [`_register_layer_hooks`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L230-L270) |
| gradient reduce-scatter | [`_make_sharded_gradient_hook`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L272-L292) |
| replicated all-reduce | [`_all_reduce_replicated_gradient`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L294-L301) |
| 执行顺序学习与预取 | [`_record_forward_use`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L316-L359) |
| observer seam | [`FSDPAllGatherEvent`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_observer.py#L13-L29) |
| adapter | [`get_fsdp`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/adapters.py#L74-L89) |

## 5. 正确性实验

### 5.1 Staff 测试

执行：

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run pytest tests/test_fsdp.py -q
```

覆盖：

1. FP32 compute；
2. FP16 compute；
3. 3-step 参数更新与完整非并行 global-batch baseline 对齐；
4. sharded weight gradient shape/dtype；
5. replicated norm gradient 跨 rank 一致。

单轮结果为 `4 passed`。最终校验按 handout 建议连续执行 5 轮。

### 5.2 扩展测试

[`OddSizedModel`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/test_fsdp_variants.py#L16-L26) 使用 35 和 15 个元素的 weight，在 2-rank 下分别形成 18 和 8 个元素的 padded shard。

测试验证：

1. full parameter gather 与分片前完全一致；
2. sharded gradient 与 parameter shard shape 一致；
3. RMSNorm gradient 存在并保持复制；
4. AdamW moments 只按本地 parameter shape 创建；
5. optimizer step 后重建参数跨 rank 一致。

### 5.3 非注册顺序执行

[`NonRegistrationOrderModel`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/test_fsdp_variants.py#L29-L39) 按 `first -> third -> second` 执行，但属性按 `first -> second -> third` 注册。[回归测试](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/test_fsdp_variants.py#L95-L122) 连续训练两步并与非并行基线逐参数对齐，验证 prefetch 使用真实执行顺序，不会在 backward 后留下孤立 all-gather。

## 6. 复杂度与通信量

设 shardable weight 总量为 $M_s$，复制小参数总量为 $M_r$，shardable layer 数为 $K$。

| 项目 | 每 rank 常驻量/调用数 |
|---|---|
| master weight | $M_s/N+M_r$ |
| gradient | $M_s/N+M_r$ |
| AdamW moments | $2M_s/N+2M_r$ |
| forward all-gather | $K$ 次 |
| backward all-gather | $K$ 次 |
| weight gradient reduce-scatter | $K$ 次 |
| replicated gradient all-reduce | 复制参数张量数次 |

按 ring collective 的每 rank 链路 volume，单次完整 weight all-gather 或 gradient reduce-scatter 约为：

$$ V_{\mathrm{collective}}\approx\frac{N-1}{N}M_s. $$

一个完整 step 的 shardable-weight 通信约含两次 all-gather 和一次 reduce-scatter：

$$ V_{\mathrm{step}}\approx3\frac{N-1}{N}M_s. $$

低精度 weight all-gather 只降低两次 all-gather 的 payload；当前 FP32 gradient reduce-scatter 不变。

## 7. 实现边界

1. 只保证静态、相同的 module 执行顺序；不支持 rank-dependent dynamic graph。
2. 不支持多个 shardable module 共享同一个 weight。
3. 只支持 dense gradient。
4. 仅对 Linear/Embedding weight 分片，其他参数保持复制。
5. 每个 weight 单独 collective，没有 flat bucket，collective latency 较高。
6. forward 有两层 lookahead；backward 预取是教学实现，不具备正式 FSDP 的 stream-aware scheduler。
7. 核心实现不保存计时数据；启用 observer 会产生少量事件分发开销。
8. 完整参数收集是同步操作，不适合训练热路径。
9. 当前没有分布式 checkpoint reshard 或 world-size 变更支持。
10. FP16 CPU 测试只验证数值路径，不代表 GPU mixed-precision 性能。

## 8. 参考资料

1. PyTorch, [`FullyShardedDataParallel`](https://docs.pytorch.org/docs/stable/fsdp.html).
2. Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277).
3. Rajbhandari et al., [ZeRO 原始论文](./references/distributed_training/zero_memory_optimizations_toward_training_trillion_parameter_models.pdf).
