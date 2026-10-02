# Flat Gradient Minimal DDP Benchmark 实验报告
## 0. 问题定义
### 0.1 Handout 要求
本报告只研究 CS336 Assignment 2 Section 5.3.1 的
`minimal_ddp_flat_benchmarking`。handout 要求修改 minimal DDP：在 backward
结束后，把所有待同步参数梯度连接为一个 flat tensor，只发起一次 batched
all-reduce；然后在与 naive minimal DDP 相同的条件下比较性能。指定条件是
`1 node x 2 GPUs` 和 Section 2.1.2 的 `xl` 模型；交付物是单次训练迭代时间、
梯度通信时间，以及 1--2 句与逐参数通信的比较。[1]

Section 2.1.2 进一步给出 `xl` 配置：

| 字段 | 值 |
|---|---:|
| $d_{\text{model}}$ | 2560 |
| $d_{\text{ff}}$ | 10240 |
| `num_layers` | 32 |
| `num_heads` | 32 |
| vocabulary size | 10000 |
| global batch size | 4 |
| context length | 512 |

这里把 handout 的 batch size 4 解释为全局 batch，再按数据并行规则均分为每
rank 2 个样本。若先前 naive DDP benchmark 实际采用了不同解释，则正式比较
必须沿用那个基线，并在结果中写清 global/local batch，不能悄悄改变工作量。[1]
### 0.2 对照方案与唯一自变量
两种方案都在完整 backward 结束后同步梯度，不与 backward 重叠：

1. **逐参数方案（naive）**：对 $K$ 个参数梯度分别执行一次同步 `SUM`
   all-reduce，再除以 world size。
2. **flat 方案**：按固定顺序把 $K$ 个梯度复制进一个连续一维 buffer，对该
   buffer 执行一次同步 `SUM` all-reduce，除以 world size，再按原 shape
   映射回各参数梯度。

比较时唯一应变化的是梯度同步方式。模型、参数 dtype、输入、loss、optimizer、
global batch、随机种子、进程拓扑、GPU、NCCL/PyTorch/CUDA 版本、warmup 和
measurement 次数都必须相同。

本题不研究 bucket 大小、梯度 ready hook、异步 collective、通信与计算重叠，
也不拿正式 `torch.nn.parallel.DistributedDataParallel` 作为被测实现。这些会
同时改变调用粒度或调度位置，无法单独回答“把 $K$ 次 collective 合成一次是否
有收益”。
### 0.3 研究问题
本报告要回答四个问题：

1. flatten/unflatten 在数学上和 PyTorch 源码中分别做了什么？
2. 合并 collective 为什么可能更快，何时 copy 成本会抵消收益？
3. flat 方案需要哪些正确性前提，如何证明它仍得到跨 rank 平均梯度？
4. 怎样测量 total step、纯 collective、完整 gradient synchronization 和额外
   峰值显存，才能得到可比较的结果？
### 0.4 符号
| 符号 | 含义 |
|---|---|
| $P$ | world size，本题 $P=2$ |
| $r$ | rank，$r\in\{0,\ldots,P-1\}$ |
| $K$ | 本轮具有 dense gradient 的唯一参数张量数量 |
| $g_j^{(r)}$ | rank $r$ 的第 $j$ 个本地参数梯度，数学上视为列向量 |
| $n_j$ | $g_j^{(r)}$ 的元素数 |
| $m_j$ | $g_j^{(r)}$ 的字节数 |
| $N$ | flat gradient 的总元素数，$N=\sum_{j=1}^{K}n_j$ |
| $M$ | 总梯度 payload 字节数，$M=\sum_{j=1}^{K}m_j$ |
| $\alpha$ | 一轮通信的固定延迟项，单位为秒 |
| $\beta$ | 每传输一字节的时间，单位为秒/字节 |

数学上梯度按列向量书写；PyTorch 参数梯度仍保留各层定义的实际 tensor shape。
flatten 只是在框架层把这些 tensor 按元素线性化为连续的一维 tensor，两种约定
并不冲突。
## 1. Flatten 与 Unflatten
### 1.1 数学定义
定义 flatten 算子 $F$ 按参数遍历顺序连接梯度：

$$f^{(r)}=F\left(g_1^{(r)},\ldots,g_K^{(r)}\right)=\begin{bmatrix}(g_1^{(r)})^\top&\cdots&(g_K^{(r)})^\top\end{bmatrix}^\top\in\mathbb{R}^{N}$$

令 $o_1=0$，$o_j=\sum_{\ell=1}^{j-1}n_\ell$。第 $j$ 个 unflatten 操作
$U_j$ 取 flat buffer 的区间 $[o_j,o_j+n_j)$，再恢复为 $g_j$ 的 shape：

$$U_j(f)=\operatorname{reshape}\left(f[o_j:o_j+n_j],\operatorname{shape}(g_j)\right)$$

这里必须保存同一组 template gradients 或等价的 shape/numel 元数据。只知道
总长度 $N$ 无法唯一恢复张量边界。
### 1.2 PyTorch helper 的源码语义
handout 建议使用 `torch._utils._flatten_dense_tensors` 和
`torch._utils._unflatten_dense_tensors`。[1] PyTorch 2.11 源码给出的契约是：

ATen 是 PyTorch 底层的 C++ 张量与算子库。Python 层的 tensor 操作最终会分派到
ATen 中对应的底层实现；这里查看 ATen 源码，是为了确认 flatten/unflatten
是否分配新存储、是否产生 copy，以及返回值是否与 flat buffer 共享存储。

1. `_flatten_dense_tensors` 假设输入为相同 dense type，返回含全部输入值的
   contiguous 1-D buffer。[2]
2. flatten 会把每个梯度按元素顺序线性化，再拼接到一个新的连续 buffer。
   这个 pack 过程可能产生设备内 copy，因此必须计入 flat 方案的时间和内存成本。[3]
3. `_unflatten_dense_tensors` 使用 template tensors 的 size 恢复输出。[2]
4. ATen 的 unflatten 会按 template shape 切分 flat buffer，并返回共享 flat
   storage 的 views；它不会自动把结果写回原 `.grad`。[3]

因此完整数据流应理解为：

```text
original .grad tensors
-> contiguous/concatenate (pack)
-> flat buffer
-> in-place SUM all-reduce
-> divide by P
-> narrow/view (unflatten metadata)
-> copy or rebind synchronized values to parameter.grad
-> optimizer.step
```

`torch._utils` 是内部 helper，不是稳定 public API。报告和实现必须记录实际
PyTorch 版本；升级版本后应重新核对源码行为，而不能把内部接口当作长期兼容
承诺。
### 1.3 Copy-back 与 rebind
unflatten 返回 view 后有两种处理：

1. **copy-back**：把每个 view `copy_` 到原 `.grad`。这保留原 gradient tensor
   的 identity、storage 和 stride，语义保守，但增加一次总量为 $M$ 的设备内
   copy。
2. **rebind**：让 `.grad` 直接引用 flat buffer 的各段 view。它可能省掉
   copy-back，但所有梯度会共享一个底层 storage，flat buffer 的生命周期至少
   延长到这些 `.grad` 被释放；还必须验证 optimizer、`zero_grad`、gradient
   hooks 和非标准 stride 对这种别名关系的兼容性。

本题的首个正确实现宜采用 copy-back，并把其时间计入“完整梯度同步时间”。
如果后续改成 rebind，应作为单独实验变量，不能与“collective 数量变化”混在
同一组比较中。
## 2. 正确性
### 2.1 Flat all-reduce 等价于逐参数 all-reduce
`torch.distributed.all_reduce` 原地归约输入 tensor，并让 process group 中每个
rank 得到归约结果；默认 reduction 是 `SUM`。[4] flatten 是按段连接的线性
重排，所以：

$$\sum_{r=0}^{P-1}f^{(r)}=F\left(\sum_{r=0}^{P-1}g_1^{(r)},\ldots,\sum_{r=0}^{P-1}g_K^{(r)}\right)$$

对 flat sum 除以 $P$ 后再取第 $j$ 段：

$$U_j\left(\frac{1}{P}\sum_{r=0}^{P-1}f^{(r)}\right)=\frac{1}{P}\sum_{r=0}^{P-1}g_j^{(r)}$$

因此在实数数学中，单次 flat `SUM` all-reduce 加一次 world-size 归一化，与对
每个参数分别执行同样操作完全等价。它改变的是消息边界，不改变每个梯度元素
参与的跨 rank 求和。
### 2.2 必须满足的前提
1. 所有 rank 以完全相同的顺序选择同一组唯一 Parameter。
2. 所有 rank 的 `grad is None` 模式相同；否则 flat 长度或段边界不一致，可能
   造成错误或 collective hang。
3. 输入 helper 的梯度必须是 dense tensor，并具有兼容的 device 和 dtype。
   `_flatten_dense_tensors` 明确假设 same dense type。[2]
4. mixed-dtype 模型通常需要按 `(device, dtype)` 分组，因而会变成每组一次
   collective，而不再是字面意义上的全模型单次 collective。
5. tied weights 必须按唯一 Parameter 只收集一次；冻结参数和本轮未使用参数
   不应伪装成需要同步的普通梯度。
6. flat buffer 必须只除以 $P$ 一次，并在 optimizer 读取 `.grad` 前完成
   unflatten/copy-back。
7. 参数 shape、numel 或遍历顺序在 wrap 后不能改变；模板顺序必须与 pack 顺序
   完全一致。

逐参数方案和 flat 方案可以因为消息大小不同而触发不同 backend 算法或协议。
浮点加法又不满足结合律，所以测试应要求数值接近，而不应无依据地要求两个方案
bitwise identical。
### 2.3 正确性验收
正式 benchmark 的完整验收目标如下；本仓库实际覆盖见本节末尾，未覆盖项属于后续增强而非已通过声明：

1. 在小模型上保存 all-reduce 前的本地梯度，离线构造逐参数跨 rank 平均值，
   与 flat 同步后的每个 `.grad` 做 `torch.testing.assert_close`。
2. 对相同初值和同一 global batch，比较 naive、flat 和单进程 global-batch
   reference 在一个 step 后及连续多个 step 后的参数。
3. 覆盖冻结参数、tied weights、空 tensor；对 sparse gradient 和 rank-dependent
   unused parameter 给出明确拒绝或实现一致协议。
4. 断言每个 rank 的 $K$、$N$、dtype/device 签名相同，并确认 flat 路径每步
   恰好调用一次 `dist.all_reduce`。
5. 分别比较 all-reduce 后、除法后和 copy-back 后的值，以便在失败时定位是
   collective、缩放还是段映射错误。

正确性实现与测试已完成。`uv run pytest tests/test_ddp_variants.py -q` 覆盖三种策略 × 两个模型，共 6 个策略/模型组合；每个组合连续两步与单进程全局 batch 基线比较。flat 路径通过普通模型、冻结参数和 tied-weight alias 检查。
## 3. 一次 Collective 与逐参数 Collective 的 Alpha-Beta 模型
### 3.1 Ring all-reduce 近似
用常见的 ring reduce-scatter 加 ring all-gather 作解释模型。每个阶段有
$P-1$ 轮，每轮发送消息的一段；对大小为 $m$ 字节的 tensor：

$$T_{\mathrm{AR}}(m,P)\approx 2(P-1)\alpha+2\frac{P-1}{P}m\beta$$

这里的 $\alpha$ 是每轮固定延迟，$\beta$ 是有效单位字节时间。handout 给出的
理想 ring 每 rank 通信量也是 $2(P-1)m/P$。[1] 真实 NCCL 会根据拓扑、消息
大小、channel 和 protocol 自动选择实现，所以该式用于解释趋势，不用于预测
精确毫秒数。
### 3.2 逐参数方案
逐参数方案有 $K$ 次 all-reduce：

$$T_{\mathrm{comm,individual}}\approx\sum_{j=1}^{K}T_{\mathrm{AR}}(m_j,P)=2K(P-1)\alpha+2\frac{P-1}{P}M\beta$$

payload 项只与总字节数 $M$ 有关，但固定延迟项被支付 $K$ 次。除此之外，Python
到 c10d、c10d 到 NCCL 的调用，collective 调度和 kernel launch 也会形成逐调用
固定成本；这些可以被视为有效 $\alpha$ 的一部分。
### 3.3 单次 flat 方案
flat 方案的网络项是一次大小为 $M$ 的 all-reduce，还要支付 pack/unpack：

$$T_{\mathrm{sync,flat}}\approx T_{\mathrm{pack}}+2(P-1)\alpha+2\frac{P-1}{P}M\beta+T_{\mathrm{unpack}}$$

于是理想差值为：

$$T_{\mathrm{comm,individual}}-T_{\mathrm{sync,flat}}\approx 2(P-1)(K-1)\alpha-T_{\mathrm{pack}}-T_{\mathrm{unpack}}$$

在本题 $P=2$ 时：

$$T_{\mathrm{comm,individual}}\approx2K\alpha+M\beta,\qquad T_{\mathrm{sync,flat}}\approx T_{\mathrm{pack}}+2\alpha+M\beta+T_{\mathrm{unpack}}$$

结论不是“flat 必然更快”，而是它用设备内存 copy 换掉约 $K-1$ 次 collective
固定成本。参数 tensor 很多且偏小时，节省的 $\alpha$ 项可能占优；若少数大
tensor 已经带宽饱和，或 flat buffer 的 pack/copy-back 很贵，收益可能缩小甚至
反转。

当前本地 `xl` 模型定义的静态审计结果是 291 个 trainable parameter tensors、
3,406,809,600 个参数。[9] 若梯度为 FP32，则 $M=13,627,238,400$ bytes，
约 13.63 GB 或 12.69 GiB；逐参数到 flat 把调用数从 291 降为 1。在 ring 模型
下，$P=2$ 时两者每 rank 的理想网络字节量仍都是 $M$，主要差异是固定调用成本
与本地 copy。该静态核算不是 benchmark 结果。
## 4. 内存分配与设备内副本
### 4.1 峰值容量
backward 后原始 `.grad` 总计已经占用约 $M$ 字节。对多个输入，
`_flatten_dense_tensors` 最终用 `cat` 创建连续 flat buffer，因此同步期间通常
还要同时保留一个约 $M$ 字节的 buffer。[2][3] PyTorch 自己的 model-averaging
工具也明确说明，flatten 为 all-reduce 效率会要求与被 flatten 参数同等大小的
额外内存。[5]

若采用 copy-back，unflatten view 本身主要增加 $O(K)$ 的 Tensor 元数据，不再
分配另一个 $M$；但 flat buffer 必须存活到所有 copy 完成。仅看 gradient
storage，峰值近似为：

$$\text{gradient-related peak}\approx M_{\mathrm{original\ grads}}+M_{\mathrm{flat}}+O(K)$$

这不包括参数、activation、optimizer state、CUDA context、NCCL workspace 和
allocator fragmentation。对当前 FP32 `xl` 静态规模，flat buffer 一项就约
12.69 GiB，所以正式运行前必须先核对每张 GPU 的可用显存。
### 4.2 最低设备内存流量
对 contiguous gradients，pack 至少读取原 gradients 的 $M$ 字节并写入 flat 的
$M$ 字节；copy-back 再读取 flat 的 $M$ 字节并写回 gradients 的 $M$ 字节。
因此 copy-back 设计引入的最低额外设备内存流量约为：

$$V_{\mathrm{pack+copyback}}\gtrsim4M$$

这里的 $4M$ 可以直接拆成四项：

| 阶段 | 读取 | 写入 | 设备内存流量 |
|---|---:|---:|---:|
| pack：原 gradients $\rightarrow$ flat buffer | $M$ | $M$ | $2M$ |
| copy-back：flat buffer $\rightarrow$ 原 gradients | $M$ | $M$ | $2M$ |
| 合计 | $2M$ | $2M$ | $4M$ |

这表示内存总线搬运了约 $4M$ 字节，不表示额外分配了 $4M$ 字节存储。额外的主要
存储仍是一个大小约为 $M$ 的 flat buffer；unflatten 只创建 views，不再复制
一份完整梯度。

这也不是网络流量。若某些 gradient 非 contiguous，`contiguous()` 还可能产生
中间 copy。flat 上的 `div_(P)` 也要读写 $M$，但 naive 的逐梯度 `div_` 总计
同样读写 $M$；二者不同的是 kernel launch 数量，而不是该操作的渐近字节量。
### 4.3 显存测量
可在 backward 完成、gradient sync 开始前调用
`torch.cuda.reset_peak_memory_stats()`，记录 `memory_allocated()` 基线，完成
sync 后读取 `max_memory_allocated()`。官方文档说明该指标返回 tensor 占用的
峰值字节数，并可用 reset 函数重新设置统计起点。[8]

应同时报告：

1. sync 前 `memory_allocated`；
2. sync 区间 `max_memory_allocated`；
3. 两者差值；
4. `memory_reserved`/`max_memory_reserved`，用于区分活跃 tensor 与 caching
   allocator 保留显存。

不要用 `nvidia-smi` 的单个快照替代区间峰值，也不要把 NCCL 或 CUDA context
等非 PyTorch tensor 内存错误归入 flat buffer。
## 5. Benchmark 方法
### 5.1 固定环境
正式报告必须记录以下信息：

| 类别 | 必填字段 |
|---|---|
| hardware | GPU 型号与数量、显存、GPU 拓扑/互联、CPU |
| software | OS、Python、PyTorch、CUDA runtime/driver、NCCL 版本 |
| distributed | 1 node、2 processes、每进程独占一张 GPU、NCCL backend |
| model | 完整 `xl` 配置、参数量、parameter tensor 数量、gradient dtype |
| workload | global/local batch、context 512、loss、optimizer、输入生成方式 |
| benchmark | 5 warmup、10 measurement、独立重复次数、随机种子 |

handout 对 distributed benchmark 明确建议：同机比较、NCCL 至少 5 次 warmup、
GPU 计时调用 `torch.cuda.synchronize()`，并汇总不同 rank 的时间。[1]
PyTorch 官方文档也说明 `torch.cuda.synchronize()` 会等待目标设备所有 stream
中的 kernel 完成。[6]
### 5.2 运行协议
1. 每个进程先设置本地 CUDA device，再初始化 NCCL process group；process
   group 初始化、模型构造、初始参数广播和随机输入创建全部排除在计时外。
2. naive 与 flat 使用同一份初始 state、同一组预先生成的 batch 序列和相同
   optimizer 配置。每种方案用独立进程启动，避免前一方案的 allocator/cache
   状态污染后一方案；运行顺序可交替以减小热环境偏差。
3. 先执行 5 个完整训练 warmup steps，其中包含实际 gradient synchronization
   和 optimizer step。warmup 不能只预热 forward/backward。
4. 正式执行 10 个 measurement steps。每个 step 在 timed region 外执行 barrier 与 device synchronize，
   建立共同起点；barrier 本身不计入 step。
5. 每个 step 记录各 rank 时间，测量后再用 `all_gather_object` 收集。以逐 step
   的最大 rank 时间作为该分布式 step 的关键路径样本，再报告 10 个样本的
   mean、population standard deviation、min 和 max。
6. 至少做 3 次独立进程级重复，保留每次完整结果；不能只挑最快一次。
### 5.3 三种计时边界
**A. Total step。** 计时范围固定为：

```text
zero_grad  # timer 之前
-> timer start -> forward
-> loss
-> backward
-> gradient synchronization
-> optimizer.step
```

计时前后都执行 CUDA synchronization，使用 host 高分辨率 timer。两种实现的
`zero_grad(set_to_none=...)` 设置必须一致。[1][6]

**B. Collective-only。** naive 从第一次 `dist.all_reduce` 前计时到最后一次
all-reduce 完成；flat 只包围那一次 flat all-reduce。区间结束必须显式 CUDA
synchronize，因为 `async_op=False` 的 Python 返回不等于 GPU 通信已经完成，
handout 对此有明确提醒。[1] 该指标回答“collective 本身省了多少”，但不包含
flat 必需的 pack/copy-back。

**C. Gradient-sync end-to-end。** naive 包含所有 all-reduce 和除法；flat 包含
pack、all-reduce、除法、unflatten 和 copy-back。handout 的“time spent
communicating gradients”应以该指标为主，否则会系统性高估 flat 的收益。

当前统一 runner 实际输出 A 和 C；B 只作为可选 profiler 诊断口径，没有写入本次结果表。这样避免为了拆出 collective-only 而改变 wrapper 或在主测量路径中增加额外同步。

为了避免区间内同步扰动 total-step 主测量，A 与 B/C 宜用相同配置的独立测量
pass。可再用 NVTX/Nsight 验证边界和 NCCL call 数，但 profiler 数据不替代未
插桩的 A 结果。

通信占比定义为：

$$\text{gradient-sync share}=100\%\times\frac{\operatorname{mean}(T_{\mathrm{sync,max-rank}})}{\operatorname{mean}(T_{\mathrm{step,max-rank}})}$$

如果另行执行 B，可报告 `collective-only` 与 `sync end-to-end` 的差，以量化 flat
pack/unpack 和缩放开销。不要把各 rank 的平均值当作用户实际等待时间，也不要
把 10 次 step 累计时间误写成单步延迟。
### 5.4 公平性与诊断检查
1. 两种方案每 step 处理的 token 数和 optimizer update 次数相同。
2. timed region 内不打印、不保存 checkpoint、不重新分配输入。
3. 记录是否启用 autocast/compile；只要一项不同，就不是本题的一变量比较。
4. 检查是否 OOM、thermal throttling 或其他进程占用 GPU；失败配置按失败报告，
   不用缩小模型后的数字冒充指定 `xl` 结果。
5. flat 若因 dtype/device 分组实际发起多次 collective，必须报告真实次数，不能
   标成“single batched all-reduce”。
6. 同时保存原始逐 step、逐 rank 样本，汇总表不能成为唯一数据来源。
## 6. 实现

### 6.1 Flat DDP 数据路径

实现位于 [`cs336_systems/ddp.py:L60-L85`](../cs336_systems/ddp.py#L60-L85)。`FlatDistributedDataParallel.finish_gradient_synchronization()` 按公共参数顺序收集本轮 dense gradients，先验证它们位于同一 device 且 dtype 相同，再执行：

```text
_flatten_dense_tensors
-> one SUM all_reduce
-> flat_gradient.div_(world_size)
-> _unflatten_dense_tensors
-> copy_ back to every original .grad
```

#### Unflatten 与 copy-back 的逐行解释

all-reduce 修改的是新创建的 `flat_gradient`，原来的各个 `parameter.grad` 并不会随之自动更新。因此同步完成后还要执行：

```python
for gradient, synchronized in zip(
    gradients,
    _unflatten_dense_tensors(flat_gradient, gradients),
    strict=True,
):
    gradient.copy_(synchronized)
```

其中两个序列分别是：

1. `gradients`：参数原本持有的 `.grad` tensor，顺序与 flatten 前完全相同。
2. `_unflatten_dense_tensors(flat_gradient, gradients)`：以原 gradients 的 shape 为模板，把 flat buffer 切成相同数量、相同 shape 的 tensor view。

例如原梯度为：

```text
gradient_1: shape=(2, 2), numel=4
gradient_2: shape=(3,),   numel=3
```

flatten 后得到长度为 7 的 buffer；unflatten 再产生：

```text
synchronized_1 = flat_gradient[0:4].view(2, 2)
synchronized_2 = flat_gradient[4:7].view(3)
```

这些 `synchronized` tensor 是 flat buffer 的视图，不是原 `.grad`。`zip()` 按原顺序把每个 `.grad` 与对应视图配对，`strict=True` 要求两边数量完全相同，避免数量不一致时被普通 `zip()` 静默截断。

最后的：

```python
gradient.copy_(synchronized)
```

把同步后的数值原地写回原 `.grad`。这里不能写成 `gradient = synchronized`，因为那只会修改循环内的局部变量，不会改变 `parameter.grad`。使用 `copy_()` 还能保留原 `.grad` 的对象 identity、storage 和 stride，使 optimizer 继续读取原有 gradient tensor，只是其中的值已经变成跨 rank 平均后的结果。

因此这段代码完成的是：

```text
flat buffer 中的同步结果
-> 按原 shape 建立 views
-> 与原 .grad 一一配对
-> 原地写回 .grad
-> optimizer.step() 读取同步后的梯度
```

本实现选择 copy-back，不改变 `.grad` 的 identity 和原 storage。sparse gradient 会显式报错；mixed dtype/device 不做隐式分组，而是显式拒绝，从而保持“每步恰好一个 flat collective”的实验变量。`module.parameters()` 默认按对象去重，因此 tied Parameter 只进入一次。

统一 runner 位于 [`cs336_systems/ddp_benchmark.py`](../cs336_systems/ddp_benchmark.py)，CLI 位于 [`scripts/benchmark_ddp.py`](../scripts/benchmark_ddp.py)。naive 与 flat 共用模型、输入、optimizer、warmup、测量和 rank-max 汇总路径，variant 是唯一实现变量。

### 6.2 正确性测试

[`tests/test_ddp_variants.py:L24-L67`](../tests/test_ddp_variants.py#L24-L67) 在 2-rank CPU/Gloo 环境中，对 naive 和 flat 两种实现分别测试 `ToyModel` 和 `ToyModelWithTiedWeights`。每个组合连续执行两步，逐步与单进程全局 batch 基线比较，并检查跨 rank 参数一致与 tied alias 未被破坏。

```bash
uv run pytest tests/test_ddp_variants.py -q
```

测试覆盖 flat 的 flatten、SUM、除法、unflatten 和 copy-back 整条数值路径；支持范围仍限定为各 rank 使用相同静态图、相同 `grad is None` 模式、dense 且同 dtype/device 的 gradients。

### 6.3 复现命令

CPU/Gloo 工程 smoke 命令为：

```bash
uv run python scripts/benchmark_ddp.py \
  --backend gloo --world-size 2 --variants naive flat \
  --model-size xl --d-model 64 --d-ff 128 --num-layers 2 --num-heads 4 \
  --vocab-size 256 --global-batch-size 4 --context-length 32 \
  --warmup-steps 3 --measurement-steps 10 --repeats 3 --num-threads 1 \
  --output-dir benchmark_results/ddp/cpu_gloo_naive_flat_20261002
```

正式双 GPU 比较命令为：

```bash
uv run python scripts/benchmark_ddp.py \
  --backend nccl --world-size 2 --variants naive flat --model-size xl \
  --vocab-size 10000 --global-batch-size 4 --context-length 512 \
  --warmup-steps 5 --measurement-steps 10 --repeats 3 \
  --output-dir benchmark_results/ddp/gpu_nccl_xl_naive_flat
```

## 7. 实验结果

### 7.1 CPU/Gloo 工程 smoke

实验日期为 2026-10-02。环境为 Intel Xeon Platinum 8336C、56 logical CPUs、Linux 5.4.143、Python 3.13.12、PyTorch 2.11.0+cu130 和 2-rank Gloo；每 rank 限制为 1 个 intra-op thread。模型为 $d=64$、$d_{\text{ff}}=128$、2 层、4 heads、$V=256$、global batch 4、context 32、FP32；每 case 为 3 warmup + 10 samples，共 3 次独立进程级 repeats。

| 实现 | calls/step | step mean ± repeat std (ms) | 完整同步 mean ± repeat std (ms) | sync share | 相对 naive 加速 |
|---|---:|---:|---:|---:|---:|
| naive per-parameter | 21 | 19.755 ± 1.131 | 12.143 ± 0.623 | 61.5% | $1.00\times$ |
| single flat buffer | 1 | 8.746 ± 0.357 | 1.746 ± 0.249 | 19.9% | $2.26\times$ |

每个数值是三次独立进程级 repeat 的 rank-max step mean 再求均值；误差是三个 repeat mean 的 population standard deviation。原始逐 rank 样本与 6 个 case JSON 位于 `benchmark_results/ddp/cpu_gloo_naive_flat_20261002/`。

在这个小型 CPU/Gloo workload 上，flat 把调用数从 21 降为 1，完整同步时间降低 85.6%，端到端 step 降低 55.7%，同步占比降低 41.6 个百分点。这与固定 collective 成本占主导的解释一致，但不能外推到 NCCL、`xl` 或不同互联。

### 7.2 正式 GPU 结果为何不可得

当前开发机没有 CUDA device；可访问的 `cuda-via-a` 只有一张 GTX 1060 6 GB，`agent1` 没有 GPU。NCCL runner 因此在 worker 启动前明确报错：`NCCL requires 2 distinct visible CUDA devices`。

对 `xl`，参数、gradient 和两个 Adam moments 的静态 FP32 下界已经是 50.765 GiB/卡。flat 同步还需一个 12.691 GiB 连续 buffer，因此仅这五份 storage 就是：

$$5\times4N_\theta=68{,}136{,}192{,}000\ \text{bytes}\approx63.457\ \text{GiB/卡}$$

这仍未包含 activation、临时张量和 runtime workspace。当前硬件既不满足两张独占 GPU，也远不满足容量要求，所以正式 `xl` 数值与 sync peak delta 记为“硬件阻塞”，不使用 CPU smoke 或理论估算冒充。

### 7.3 Handout 要求的比较结论

工程 smoke 中，单 flat buffer 相比逐参数 naive 将完整梯度同步均值从 12.143 ms 降到 1.746 ms，并将 step 均值从 19.755 ms 降到 8.746 ms；该结果说明本 workload 的 collective 固定成本明显高于 pack/copy-back 成本。正式 `2×GPU + NCCL + xl` 的方向和幅度仍需在至少约 64 GiB/卡并留有 activation 余量的双 GPU 节点复跑后确认。

## 8. 结论

1. flat all-reduce 在数学上与逐参数平均等价，前提是各 rank 的参数顺序、gradient 存在性、dtype、device 和 shape 一致。
2. 实现使用一个连续 buffer、一次 SUM all-reduce、一次 world-size 归一化和 copy-back，并通过普通模型与 tied-weight 模型的 2-rank 基线测试。
3. CPU/Gloo smoke 的 6/6 case 全部成功，flat 在该 workload 上得到 $2.26\times$ 端到端加速。
4. flat 用 $O(M)$ 额外存储和约 $4M$ pack/copy-back 内存流量换取 collective 数从 $K$ 降到 1；在 `xl` FP32 下，额外 buffer 本身约 12.691 GiB。
5. 正式 GPU/NCCL 实验受硬件数量与容量阻塞，报告保留真实复跑命令和边界，不提供虚构值。
## 9. 一手来源
1. [Stanford CS336 Spring 2026 Assignment 2 handout](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)
2. [PyTorch 2.11 `torch._utils` flatten/unflatten Python wrappers](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_utils.py#L568-L618)
3. [PyTorch 2.11 ATen `flatten_dense_tensors` / `unflatten_dense_tensors`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/TensorShape.cpp#L4791-L4820)
4. [PyTorch 2.11 `torch.distributed.all_reduce`](https://docs.pytorch.org/docs/2.11/distributed.html#torch.distributed.all_reduce)
5. [PyTorch 2.11 model averaging utility and flat-buffer memory note](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/algorithms/model_averaging/utils.py#L21-L49)
6. [PyTorch 2.11 `torch.cuda.synchronize`](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html)
7. [PyTorch 2.11 `torch.cuda.Event`](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html)
8. [PyTorch 2.11 CUDA peak-memory APIs](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html)
9. [本仓库 `TransformerLM` 一手实现](../cs336-basics/cs336_basics/model.py)
