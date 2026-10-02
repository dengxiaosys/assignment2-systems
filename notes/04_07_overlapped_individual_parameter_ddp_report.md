# 逐参数通信与反向计算重叠的 DDP 实验报告

> 研究对象：CS336 Assignment 2 `ddp_overlap_individual_parameters`
> 实验日期：2026-10-01
> 状态：实现与 CPU/Gloo 正确性实验完成；GPU overlap 性能属于后续实验

## 0. 实验要解决什么问题

naive DDP 的执行顺序是：

```text
forward
-> backward 计算全部参数梯度
-> 逐参数 all-reduce
-> optimizer.step
```

所有通信都发生在 backward 结束后，因此完整通信时间都暴露在训练 step 的关键路径上。

本实验只改变通信的发起时机：

```text
某个参数的梯度 ready
-> 立即异步发起该梯度的 all-reduce
-> backward 继续计算其他参数梯度
-> backward 结束后等待全部通信
-> optimizer.step
```

实验需要回答两个问题：

1. 异步化以后，梯度平均结果是否仍与全局 batch 基线一致？
2. 实现是否保证 optimizer 只能在全部异步通信可安全消费后运行？

本实验不测量 GPU overlap。`async_op=True` 只创造重叠机会，是否真的发生 GPU kernel overlap 必须由后续 Nsight benchmark 证明。[1]

## 1. 理论

### 1.1 数学目标没有改变

设 world size 为 $P$，rank $r$ 上参数 $\theta_j$ 的本地梯度为 $g_j^{(r)}$。同步后每个 rank 都应得到：

$$\bar g_j=\frac{1}{P}\sum_{r=0}^{P-1}g_j^{(r)}$$

同步和异步方案的数学目标完全相同。区别只在于：

- naive：等 backward 全部结束后再通信；
- overlap：某个梯度 ready 后立即发起通信。

因此异步化不能改变参与通信的参数、collective 顺序、归约操作或 world-size 平均。

### 1.2 梯度什么时候算 ready

参数梯度可能由多条反向路径共同贡献。只有这些贡献已经全部累积到叶子参数的 `.grad` 后，才能对该梯度发起 all-reduce。

`register_post_accumulate_grad_hook` 正好在叶子参数完成 `.grad` 累积后执行，因此 hook 中读取的 `parameter.grad` 是本次 backward 的完整本地梯度。[2]

### 1.3 为什么能够和 backward 重叠

以 forward 顺序 $L_1\to L_2\to L_3$ 为例，backward 通常反向执行：

```text
backward(L3) -> g3 ready -> launch all_reduce(g3)
backward(L2) -> g2 ready -> launch all_reduce(g2)
backward(L1) -> g1 ready -> launch all_reduce(g1)
                                |
                                +-> wait all -> optimizer.step
```

较早 ready 的 $g_3$ 可以在 $L_2$、$L_1$ 的 backward 继续执行时通信。最后 ready 的梯度后面没有多少计算可用于隐藏通信，因此通常仍会留下 communication tail。

### 1.4 异步通信的依赖关系

`dist.all_reduce(..., async_op=True)` 会立即返回一个 `Work` handle。Python 调用返回不代表通信结果已经可以被 optimizer 使用。

正确依赖关系是：

```text
AccumulateGrad 更新 parameter.grad
-> hook 发起 async all-reduce
-> 保存 Work
-> backward 返回
-> wait 每个 Work
-> gradient / world_size
-> optimizer.step
```

如果漏掉 `wait()`，optimizer 可能在通信仍读写 `.grad` 时读取该梯度，造成数据竞争或错误更新。[3]

### 1.5 通信工作量

设有 $K$ 个梯度 tensor，总字节数为 $M$。逐参数 overlap 仍然发起 $K$ 次 collective：

$$T_{\mathrm{comm}}\approx K\alpha+\beta M$$

它没有减少通信次数或 payload，只尝试把部分通信时间隐藏到 backward 计算中。减少 collective 次数属于上一题 flat-gradient DDP；本题只研究调度时机。

## 2. 实验实现

### 2.1 代码位置与职责

| 位置 | 职责 |
|---|---|
| [`cs336_systems/ddp.py:L88-L125`](../cs336_systems/ddp.py#L88-L125) | overlap wrapper、hook、异步 Work 和 finish |
| [`tests/adapters.py:L37-L69`](../tests/adapters.py#L37-L69) | 把作业接口映射到 overlap wrapper |
| [`tests/test_ddp.py`](../tests/test_ddp.py) | 通过 adapter 验证五步训练等价性 |
| [`tests/test_ddp_variants.py`](../tests/test_ddp_variants.py) | 同时回归 naive、flat、overlap 三种实现 |

### 2.2 wrapper 保存的状态

`OverlappedDistributedDataParallel` 保存三类状态：

```python
self._pending_reductions: list[tuple[dist.Work, Tensor]] = []
self._pending_gradient_ids: set[int] = set()
self._hook_handles: list[RemovableHandle] = [...]
```

| 状态 | 作用 |
|---|---|
| `_pending_reductions` | 保存每个异步 Work 及其原地输出 gradient，供 finish 等待和平均 |
| `_pending_gradient_ids` | 防止同一个 `.grad` 在前一次通信完成前再次进入 hook |
| `_hook_handles` | 持有 hook 注册句柄，使 hook 生命周期与 wrapper 一致 |

### 2.3 一次训练 step 的状态变化

```text
step 开始
-> pending 为空
-> forward
-> backward
   -> parameter A ready: pending += (work_A, grad_A)
   -> parameter B ready: pending += (work_B, grad_B)
   -> ...
-> finish
   -> wait work_A; grad_A /= P
   -> wait work_B; grad_B /= P
   -> ...
   -> 清空 pending
-> optimizer.step
```

下一次 forward 前，pending 必须已经清空。

## 3. 关键代码解释

### 3.1 注册 post-accumulate hook

```python
self._hook_handles = [
    parameter.register_post_accumulate_grad_hook(self._all_reduce_gradient)
    for parameter in self._parameters_to_sync
    if parameter.requires_grad
]
```

这段代码：

1. 遍历唯一 Parameter；
2. 跳过 frozen parameter；
3. 在每个可训练叶子参数上注册 hook；
4. 保存 `RemovableHandle`，避免失去 hook 生命周期管理入口。

使用 `module.parameters()` 得到的同步清单默认按 Parameter 对象去重，所以 tied weight 只注册一个 hook。

### 3.2 hook 中发起异步 all-reduce

```python
def _all_reduce_gradient(self, parameter: Tensor) -> None:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError(...)
    if gradient.is_sparse:
        raise RuntimeError(...)

    gradient_id = id(gradient)
    if gradient_id in self._pending_gradient_ids:
        raise RuntimeError(...)

    work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
    self._pending_reductions.append((work, gradient))
    self._pending_gradient_ids.add(gradient_id)
```

关键点：

- all-reduce 原地修改 `.grad`；
- `async_op=True` 让 hook 不必等待通信完成；
- `(work, gradient)` 必须共同保存，否则 finish 无法知道要等待哪个操作、平均哪个 tensor；
- `gradient_id` 用对象身份标记正在通信的 `.grad`，禁止通信未完成时再次写同一个 gradient；
- sparse gradient 不在本实验支持范围内，因此显式拒绝。

标准调用顺序是：

```text
一次 backward -> 一次 finish -> optimizer.step -> 下一次 forward
```

本实现不支持多次 backward 后再统一 finish 的梯度累积方式。

### 3.3 finish 等待并完成平均

```python
def finish_gradient_synchronization(self) -> None:
    try:
        for work, gradient in self._pending_reductions:
            work.wait()
            gradient.div_(self._world_size)
    finally:
        self._pending_reductions.clear()
        self._pending_gradient_ids.clear()
```

每个梯度先 `wait()`，再除以 world size。不能在 wait 前原地执行 `div_()`，因为通信后端可能仍在读写同一个 tensor。

`finally` 保证 Python 状态被清理。但如果通信本身失败，训练进程应终止，不能把“状态已清理”理解为可以继续训练。

### 3.4 forward guard

```python
def forward(self, *inputs, **kwargs):
    if self._pending_reductions:
        raise RuntimeError(...)
    return super().forward(*inputs, **kwargs)
```

如果上一轮通信还没有 finish，就拒绝开始下一轮 forward。这把训练状态机显式限制为：

```text
forward -> backward -> finish -> optimizer.step -> next forward
```

## 4. 实验方法

### 4.1 为什么使用 CPU/Gloo

本题验证数值正确性和异步 Work 管理，不要求测量 GPU 性能。CPU/Gloo 已足够验证：

- 参数初始化广播；
- hook 是否触发；
- all-reduce 与 world-size 平均；
- finish 是否在 optimizer 前完成；
- 多 rank 更新是否与全局 batch 基线一致。

CPU/Gloo 不能证明 CUDA/NCCL kernel overlap，因此本报告不提供性能结论或 profiler 结论。

### 4.2 测试矩阵

| 测试 | 进程/后端 | 覆盖内容 |
|---|---|---|
| [`tests/test_ddp.py`](../tests/test_ddp.py) | 2 ranks / Gloo | adapter、不同 rank 初值、5 步全局 batch 等价、frozen parameter、tied weight |
| [`tests/test_ddp_variants.py`](../tests/test_ddp_variants.py) | 2 ranks / Gloo | naive、flat、overlap × 普通/tied 模型，各连续 2 步 |

测试中的单进程基线处理完整 global batch；两个 DDP rank 各处理互不重叠的半批，并使用 mean loss。每次 optimizer step 后检查：

1. 各 rank 模型状态一致；
2. rank 0 的 DDP 参数与全局 batch 基线 `torch.allclose`；
3. frozen parameter 保持不变；
4. tied-weight alias 没有被 wrapper 破坏。

执行命令：

```bash
uv run pytest tests/test_ddp.py tests/test_ddp_variants.py -q
```

为检查异步通信是否偶发 hang，完整命令独立执行 5 次。

## 5. 实验结果

### 5.1 稳定性

| 项目 | 结果 |
|---|---:|
| 独立运行次数 | 5 |
| 每轮 pytest cases | 3 |
| 累计结果 | 15/15 passed |
| pytest 内部耗时均值 ± sample std | 12.054 ± 0.626 s |
| 进程 wall time 均值 ± sample std | 14.626 ± 0.651 s |
| hang 或数值失败 | 0 |

部分运行出现 `localhost:12390` 的 c10d socket retry warning，但 process group 随后正常建立，所有断言通过。该 warning 不表示梯度同步失败。

### 5.2 正确性

| 检查项 | 结果 |
|---|---|
| rank 0 初始状态广播到其他 rank | 通过 |
| overlap 更新与单进程 global batch 基线一致 | 通过 |
| optimizer step 后各 rank 状态一致 | 通过 |
| frozen parameter 不更新 | 通过 |
| tied weight 只注册和同步一次 | 通过 |
| 连续多轮运行无 hang | 通过 |

这些结果证明了实现的数值路径和状态管理正确，但不证明 GPU 上已经发生通信与计算重叠。

## 6. 实现边界

当前实现是课程题目的最小 overlap 版本，支持范围是：所有 rank 使用相同静态计算图、产生相同 used-parameter 集合，并以相同顺序触发 hooks。

不支持或未验证的情况：

1. **rank-dependent unused parameter**：不同 rank 缺少不同 collective，可能 hang 或错配。
2. **不同 hook ready 顺序**：collective shape 相同也可能静默归约错参数。
3. **sparse gradient**：实现显式报错。
4. **多次 backward 后统一 finish**：可能与尚未完成的通信并发写 `.grad`。
5. **逐 forward buffer 同步**：当前只在 wrapper 构造时广播 buffer。
6. **reentrant backward 和 activation checkpointing 特殊图**：未纳入测试。

正式 PyTorch DDP Reducer 会按固定 bucket 顺序提交 collective，并提供 unused-parameter 协议。本实现直接在 hook 中提交，代码更容易理解、启动通信更早，但只承诺上述静态图范围。[4]

## 7. 结论

1. post-accumulate hook 在参数 `.grad` 完成本地累积后立即发起异步 SUM all-reduce。
2. pending Work 和 gradient 必须共同保存，并在 optimizer step 前逐个 wait、再除以 world size。
3. forward guard 和 `_pending_gradient_ids` 防止上一轮通信未完成时开始下一轮或再次写同一个 `.grad`。
4. 2-rank CPU/Gloo 测试连续 5 轮、累计 15/15 cases 通过，验证了数值等价和状态管理。
5. 是否真的获得 GPU overlap 与端到端加速不属于本实验结论，留给后续 benchmark。

## 参考资料

1. [Stanford CS336 Assignment 2 Systems handout，Section 5.3.2](../cs336_assignment2_systems.pdf#page=34)。
2. [PyTorch `Tensor.register_post_accumulate_grad_hook`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html)。
3. [PyTorch Distributed：异步 collective 与 `Work`](https://docs.pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations)。
4. [PyTorch DDP Design Note](https://docs.pytorch.org/docs/stable/notes/ddp.html#internal-design)。
5. [PyTorch `DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
6. [本作业 `tests/test_ddp.py`](../tests/test_ddp.py)、[`tests/test_ddp_variants.py`](../tests/test_ddp_variants.py) 与 [`tests/common.py`](../tests/common.py)。
