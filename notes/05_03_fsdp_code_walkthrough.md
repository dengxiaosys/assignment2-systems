# `fsdp.py` 代码阅读指南

本文按一次训练 step 的执行顺序阅读 [`cs336_systems/fsdp.py`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py)，重点回答以下问题：

1. 哪些状态长期驻留，哪些张量只短暂存在；
2. 同一个 `Parameter` 为什么能在 shard shape 和完整 weight shape 之间切换；
3. forward 和 backward 分别在何时 all-gather；
4. 完整 gradient 如何变成本地 gradient shard；
5. 为什么要记录真实执行顺序；
6. 为什么计时和 accounting 不放在 FSDP 主链路中。

## 1. 先建立整体模型

不要从文件第一行开始逐句阅读。建议先记住一条主线：

```text
初始化:
  广播完整模型
  -> 每个 weight 切成 FP32 master shard

forward:
  all-gather 当前层完整 weight
  -> 执行当前层
  -> reshard

backward:
  再次 all-gather 当前层完整 weight
  -> autograd 生成完整 weight gradient
  -> reduce-scatter 为本地 gradient shard
  -> reshard

optimizer:
  用本地 gradient shard 更新本地 FP32 master shard
```

FSDP 并没有改变线性层的数学计算。数学上仍以列向量记 $y=Wx$；变化的是 $W$ 在不同执行阶段的物理存储形式。

## 2. 从外部 interface 开始

### 2.1 正常训练只需要三个操作

测试 adapter 展示了调用方真正需要知道的 interface：

1. [`get_fsdp`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/adapters.py#L74-L89)：包装模型；
2. 正常调用 `fsdp_model(inputs)` 和 `loss.backward()`；
3. [`fsdp_on_after_backward`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/adapters.py#L92-L106)：等待梯度通信，再执行 `optimizer.step()`。

典型调用顺序是：

```python
fsdp_model = get_fsdp(model, compute_dtype=torch.float16)
optimizer = torch.optim.AdamW(fsdp_model.parameters())

optimizer.zero_grad(set_to_none=True)
logits = fsdp_model(input_ids)
loss = loss_fn(logits, targets)
loss.backward()
fsdp_on_after_backward(fsdp_model, optimizer)
optimizer.step()
```

optimizer 应在 FSDP 包装完成后创建。此时 shardable `Parameter.data` 已经变成本地 shard，optimizer state 也会按 shard shape 初始化。

### 2.2 调用方必须遵守的约束

- 必须先初始化 `torch.distributed` process group；
- master weight 必须是 FP32；
- 所有 rank 必须执行相同的模型图和 collective 顺序；
- 每个 shardable module 每轮 forward/backward 各执行一次；
- `loss.backward()` 后、下一次 forward 前必须调用 `finish_gradient_synchronization()`；
- 不支持 shared weight、sparse gradient、动态执行顺序和 activation checkpoint 重计算。

这些约束也是这个教学版 FSDP interface 的组成部分，而不只是实现细节。

## 3. 文件中的三层职责

### 3.1 单个 weight 的状态：`_ShardedWeight`

[`_ShardedWeight`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L33-L119) 管理一个 `Linear.weight` 或 `Embedding.weight`：

- 原始 shape 和元素数；
- padding 后的 shard 大小；
- 当前 rank 的 FP32 master shard；
- 正在进行的异步 all-gather；
- 当前是否已经安装完整 compute weight。

它只关心一个参数，不负责整个模型的执行顺序。

### 3.2 整个模型的调度：`FullyShardedDataParallel`

[`FullyShardedDataParallel`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L122-L393) 负责：

- 找出需要分片的 weight；
- 注册 forward 和 gradient hooks；
- 调度 forward/backward all-gather；
- 启动 gradient reduce-scatter；
- 等待通信并把 gradient shard 交给 optimizer；
- 验证每轮执行顺序保持稳定。

### 3.3 可选观测 seam

[`FSDPAllGatherEvent`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp_observer.py#L13-L23) 只描述一次 all-gather 的生命周期事件。核心 FSDP 不保存耗时、ready 状态或通信统计；需要实验数据时，由外部 observer adapter 消费这些事件。

因此删除 accounting 代码不会改变 FSDP 的训练语义，而删除 `_ShardedWeight` 或 hook 调度则会使分片训练能力消失。

## 4. 第一遍阅读：参数如何被分片

从 [`_ShardedWeight.__init__`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L36-L65) 开始。

设完整 weight 展平后有 $P$ 个元素，world size 为 $N$：

$$ P_{\mathrm{shard}}=\left\lceil\frac{P}{N}\right\rceil,\qquad P_{\mathrm{padded}}=NP_{\mathrm{shard}}. $$

例如 $P=7$、$N=2$：

```text
original: [w0, w1, w2, w3, w4, w5, w6]
padded:   [w0, w1, w2, w3, w4, w5, w6, 0]
rank 0:   [w0, w1, w2, w3]
rank 1:   [w4, w5, w6, 0]
```

关键语句是：

```python
parameter.data = padded_weight.narrow(0, start, self.shard_numel).clone()
self.master_shard = parameter.data
```

`Parameter` 对象本身没有替换，所以 module、optimizer 和 hooks 仍引用同一个对象；改变的是它当前指向的 storage 和 shape。

### 4.1 为什么先广播再切分

[`_broadcast_initial_state`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L200-L204) 在 [`_build_sharded_weights`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L206-L228) 之前执行。

如果各 rank 的随机初始化不同，直接切分会把不同模型的片段拼在一起。先以 rank 0 为准广播完整参数，再切 shard，才能保证之后 all-gather 重建的是同一个模型。

### 4.2 哪些参数会分片

`_build_sharded_weights` 递归遍历所有子模块，并匹配：

```python
(Linear, Embedding, nn.Linear, nn.Embedding)
```

因此会分片：

- token embedding；
- 每层 attention 的 Q、K、V、output projection；
- 每层 FFN 的 `w1`、`w2`、`w3`；
- `lm_head`；
- 测试模型里的 PyTorch `nn.Linear.weight` 和 `nn.Embedding.weight`。

RMSNorm weight 和 `nn.Linear.bias` 不在这个集合中，保持完整复制。

## 5. 第二遍阅读：一次 all-gather

### 5.1 启动异步通信

[`start_all_gather`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L71-L84) 完成三件事：

1. 将 FP32 master shard 转成 `compute_dtype`；
2. 为每个 rank 的输出 shard 分配 buffer；
3. 调用 `dist.all_gather(..., async_op=True)`。

`_PendingAllGather.local_compute_shard` 看起来没有参与后续拼接，但不能删除：异步通信结束前必须保持发送 tensor 存活。

### 5.2 等待并安装完整 weight

[`materialize`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L86-L98)：

1. 等待 `Work`；
2. 按 rank 顺序拼接 shards；
3. 去除尾部 padding；
4. reshape 回原始 weight shape；
5. 令 `parameter.data` 指向完整 compute weight。

此时 master shard 仍由 `self.master_shard` 持有，并没有丢失。

### 5.3 恢复 shard

[`reshard`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L100-L102) 只需要重新赋值：

```python
self.parameter.data = self.master_shard
```

因为 optimizer 始终持有同一个 `Parameter` 对象，所以恢复后它看到的正是本地 FP32 shard。

## 6. 第三遍阅读：构造函数

阅读 [`FullyShardedDataParallel.__init__`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L130-L160) 时，可以把字段分为四组：

| 状态 | 作用 |
|---|---|
| `_rank`、`_world_size` | 决定 shard 所属区间 |
| `_sharded_weights`、`_parameter_to_sharded_weight` | 参数级运行时状态及反向索引 |
| `_pending_gradient_reductions`、`_pending_replicated_reductions` | 尚未等待完成的梯度 collective |
| forward/backward order 字段 | 保存真实执行顺序并驱动后续预取 |
| `_observer` | 可选观测 seam，默认不产生统计 |

初始化顺序不能随意调整：

```text
broadcast 完整状态
-> 创建 master shards
-> 建立 Parameter 到 shard state 的映射
-> 注册 layer hooks
-> 注册 gradient hooks
```

## 7. 第四遍阅读：Forward 路径

入口是 [`forward`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L162-L172)。

### 7.1 首轮 forward

首轮尚不知道真实执行顺序，因此 `_prefetch_forward_position(0/1)` 不做任何事。

每个 shardable module 执行前，[forward pre-hook](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L244-L249)：

1. 记录当前 module index；
2. 按需启动 all-gather；
3. 等待并 materialize 完整 weight。

module 执行后，[forward post-hook](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L251-L260)：

1. 在 output Tensor 上注册 backward hook；
2. 立即 reshard 当前 weight；
3. 首轮暂不预取未知的未来层。

wrapper forward 返回前，[`_finish_forward_order`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L322-L330) 保存并校验真实顺序。

### 7.2 后续 forward

执行顺序已知后：

```text
forward 开始:
  预取 position 0 和 1

position i pre-hook:
  等待 weight i

position i compute:
  与 weight i+1 的 all-gather 重叠

position i post-hook:
  reshard weight i
  启动 weight i+2 的 all-gather
```

代码使用“真实执行位置”而不是 `named_modules()` 注册序号，因为 SwiGLU 的注册顺序可能是 `w1, w2, w3`，实际执行顺序却是 `w1, w3, w2`。

## 8. 第五遍阅读：Backward 路径

Backward 分成“恢复完整 weight”和“处理完整 gradient”两个时点。

### 8.1 Output Tensor hook：在模块 backward 前恢复 weight

[`_make_output_gradient_hook`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L262-L270) 注册在 forward output 上。当梯度传播到该 output 时：

1. 记录真实 backward 顺序；
2. materialize 当前层完整 weight；
3. 按已知 backward 顺序预取后续一层和两层；
4. 原样返回 `output_gradient`。

以线性层 $y=Wx$ 为例，计算 $\nabla_xL=W^\top\nabla_yL$ 需要完整 $W$，因此必须在该层 backward 真正执行前恢复 weight。

### 8.2 Parameter post-accumulate hook：梯度完成后立即分片

当 autograd 已经把完整 weight gradient 写入 `parameter.grad` 后，[`_make_sharded_gradient_hook`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L272-L292)：

1. flatten 完整 gradient；
2. 按 weight 的 padded size 补零；
3. 分配一个 shard 大小的输出；
4. 异步启动 `reduce_scatter_tensor(..., SUM)`；
5. 清除完整 `parameter.grad`；
6. 恢复 `parameter.data = master_shard`；
7. 保存 `Work`、输出 shard 和目标 `Parameter`。

第 $r$ 个 rank 最终需要的梯度为：

$$ g^{(r)}=\frac{1}{N}\operatorname{reduce\text{-}scatter}\left(\widetilde g^{(0)},\ldots,\widetilde g^{(N-1)}\right)_r. $$

这里的 local gradient shard 已经聚合所有 rank 的 local-batch 梯度。“local”表示它对应本 rank 持有的参数区间，不表示梯度只来自本 rank 的样本。

### 8.3 复制参数使用 all-reduce

RMSNorm weight 等参数没有分片，所以它们仍需要完整平均梯度。[`_all_reduce_replicated_gradient`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L294-L301) 对这些梯度启动异步 all-reduce。

### 8.4 Finish 阶段

[`finish_gradient_synchronization`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L174-L189) 必须在 `optimizer.step()` 前执行：

1. 验证本轮 backward 顺序；
2. 等待所有 reduce-scatter；
3. 除以 world size，得到 global-batch 平均梯度；
4. 把 reduced shard 安装为 `parameter.grad`；
5. 等待并平均复制参数的 all-reduce；
6. 清理 pending 状态。

完成后，对每个 shardable weight 都有：

```text
parameter.data.shape == local master shard shape
parameter.grad.shape == local gradient shard shape
```

因此普通 AdamW/SGD 可以直接更新，不需要理解分布式通信。

## 9. 为什么 forward 与 backward 顺序要分别学习

模块注册顺序、forward 执行顺序和 backward 执行顺序不是同一个概念。

以 SwiGLU 为例：

```text
注册顺序:  w1 -> w2 -> w3
forward:   w1 -> w3 -> w2
backward:  由 autograd 图决定，不能直接用注册序号递减代替
```

[`_record_forward_use`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L316-L320) 和 [`_record_backward_use`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L332-L337) 分别记录两条序列。

如果按注册序号预取，可能在某个 weight 已经完成 backward 后再次启动它的 all-gather。这个 collective 没有消费者，会留下额外 buffer；下一轮还可能使用 optimizer step 前启动的旧 weight。对应回归测试见 [`test_fsdp_prefetch_uses_observed_execution_order`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/tests/test_fsdp_variants.py#L95-L124)。

## 10. Weight 与 activation 的内存生命周期

### 10.1 Weight 可以重新构造

完整 weight 在 forward 后可以释放，因为 optimizer 尚未更新 master shards。Backward 前再次 all-gather，可以无损得到相同的完整 weight。

### 10.2 Activation 默认必须保留

计算 $\nabla_WL=(\nabla_yL)x^\top$ 需要 forward activation $x$。FSDP 不会替代 autograd 保存这些 activation，因此它只解决 parameter、gradient 和 optimizer state 的冗余，不自动解决 activation memory。

### 10.3 瞬时峰值

parameter post-accumulate hook 开始时，一个 rank 可能短暂同时持有：

```text
当前层完整 compute weight
+ 当前层完整 weight gradient
+ padded gradient 通信输入
+ 本地 FP32 master shard
+ reduce-scatter 输出 shard
+ 已预取 weight 的通信 buffer
```

这些是逐层临时对象，而不是整个模型的完整 weight/gradient 常驻副本。静态公式 $M/N+M/N+2M/N$ 不包含这些 buffer，也不包含 activation。

### 10.4 Activation checkpointing

Activation checkpointing 只保存 block 边界，在 backward 时重算 block 内部 forward，以计算换内存。它与 FSDP 是两种独立技术，而且重算可能再次触发 weight all-gather。

当前实现要求每个 shardable module 每轮 forward/backward 各执行一次，因此不支持 checkpoint 重计算。支持它需要把一次训练 step 内的“原始 forward”和“backward recompute forward”建模为不同阶段。

## 11. 完整参数重建

[`gather_full_parameters`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L191-L198) 逐个 all-gather FP32 master shards，裁掉 padding 后恢复原 shape；复制参数直接 clone。

它适用于：

- 与非并行基线比较；
- 导出完整模型；
- correctness 检查。

它会让返回的字典同时持有整个模型的完整参数，因此不应放在训练热路径中。参数重建本身是 FSDP 状态读取能力；具体的误差计算和 diff 逻辑位于测试，不在核心实现中。

## 12. Observer 为什么独立

核心类只调用 [`_notify_all_gather`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/fsdp.py#L369-L388)，发出：

- `launched`：异步 all-gather 已启动；
- `waiting`：当前层即将消费完整 weight；
- `finished`：等待结束并完成 materialize。

默认 `observer=None`，因此核心类：

- 不调用 `perf_counter_ns()`；
- 不执行 `Work.is_completed()`；
- 不保存时间序列；
- 不计算通信字节；
- 不包含实验结果汇总。

这条 seam 允许 accounting adapter 增加诊断行为，而无需修改分片和 collective 的正确性路径。

## 13. 建议阅读顺序

第一次阅读只跟数据形状：

1. `_ShardedWeight.__init__`；
2. `start_all_gather`；
3. `materialize`；
4. `reshard`；
5. `padded_flat_gradient`。

第二次阅读只跟一次 step：

1. `FullyShardedDataParallel.forward`；
2. `_make_forward_pre_hook`；
3. `_make_forward_post_hook`；
4. `_make_output_gradient_hook`；
5. `_make_sharded_gradient_hook`；
6. `finish_gradient_synchronization`；
7. `optimizer.step()`。

第三次阅读再看调度与辅助能力：

1. `_record_forward_use` 和 `_record_backward_use`；
2. `_prefetch_forward_position` 和 `_prefetch_backward_position`；
3. `_start_all_gather` 和 `_notify_all_gather`；
4. `gather_full_parameters`；
5. `FSDPObserver`。

## 14. 调试时建议观察的值

在两 rank 小模型上逐层观察：

| 时点 | `parameter.data.shape` | `parameter.grad` | `pending_all_gather` |
|---|---|---|---|
| 初始化完成 | shard shape | `None` | `None` |
| all-gather 启动后 | shard shape | `None` | 非空 |
| forward/backward compute | full shape | 通常尚未完成 | `None` |
| gradient hook 开始 | full shape | full gradient | `None` |
| gradient hook 结束 | shard shape | `None` | `None` |
| finish 完成 | shard shape | gradient shard | `None` |
| optimizer step 后 | shard shape | gradient shard 或清零状态 | `None` |

最重要的断言不是某个临时对象是否存在，而是：

```text
进入层计算前：parameter.data 是完整 weight
离开层计算后：parameter.data 是 master shard
optimizer.step 前：parameter.grad 是匹配 master shard 的全局平均 gradient shard
下一轮 forward 前：不存在未完成的 gradient collective
```

掌握这四条不变量后，文件中的 hook 和 pending-work 管理会容易理解很多。
