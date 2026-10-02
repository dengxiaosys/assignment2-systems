# Naive DDP 正确性实验报告

## 0. 问题定义

### 0.1 研究对象

本报告对应 CS336 Assignment 2 handout 的 Section 5.2 `naive_ddp`。目标是在数据并行训练中让 $P$ 个进程各自持有完整模型副本，各处理互不重叠的本地 batch，在 backward 完成后逐参数同步梯度，再由每个进程独立执行相同的 optimizer step。handout 要求实现 `adapters.get_ddp` 和可选的 `adapters.ddp_on_after_backward`，并用 `uv run pytest tests/test_ddp.py` 验证。[1]

这里的 “naive” 特指以下两个调度选择：

1. 一个参数张量对应一次 collective，不做梯度 flatten 或 bucket 合并。
2. 等整个 backward 结束后才开始同步梯度，不把通信与 backward 计算重叠。

这两个选择正是 handout 后续优化章节要解决的局限：合并小 collective 以降低逐调用开销，以及在梯度就绪时异步发起通信以隐藏通信时间。[1]

### 0.2 要验证的正确性命题

一次 naive DDP 训练迭代应满足：

1. 初始化完成后，所有 rank 的模型参数相同。
2. 每个 rank 只处理全局 batch 的一个互不重叠分片。
3. 对每个有梯度的唯一参数张量，所有 rank 最终持有相同的平均梯度。
4. 在参数、optimizer 配置、optimizer state 和平均梯度都相同时，各 rank 独立执行 optimizer step 后仍保持参数与 optimizer state 同步。
5. 当全局 batch 被均匀切分且每个 rank 使用 mean loss 时，一次 DDP 更新与单进程在全局 batch 上的更新等价。[1][5][9]

### 0.3 符号与约定

| 符号 | 含义 |
|---|---|
| $P$ | process group 的 world size |
| $r$ | rank，$r\in\{0,\ldots,P-1\}$ |
| $B$ | 每个 rank 的本地有效 loss 元素数 |
| $\theta_t$ | 第 $t$ 次更新前的模型参数列向量 |
| $g_t^{(r)}$ | rank $r$ 在第 $t$ 次更新中得到的本地梯度列向量 |
| $\bar g_t$ | 跨 rank 平均后的梯度列向量 |
| $s_t$ | optimizer state；例如 momentum、Adam 一阶矩、二阶矩和 step 计数 |
| $K$ | 有梯度的唯一参数张量数量 |
| $m_j$ | 第 $j$ 个参数梯度的字节数 |
| $M$ | 一次更新中全部唯一参数梯度的总字节数，$M=\sum_{j=1}^{K}m_j$ |

数学中的参数和梯度均按列向量书写。PyTorch 参数 tensor 的实际形状由各层定义，框架通常把 batch 放在前面、特征放在最后一维；这与数学上的列向量约定属于两个不同层面。

## 1. Naive DDP 的算法流程

### 1.1 进程组与模型副本

每个进程先初始化同一个 process group，并在本 rank 的目标 device 上构造完整模型。`torch.distributed` 的 collective 作用于 process group；默认 group 覆盖整个 world。单机多 GPU 通常采用一进程一 GPU，并保证每个 NCCL 进程独占其使用的 GPU。[4][5]

DDP 不负责自动切分输入；调用方必须让不同 rank 读取互不重叠的数据分片。PyTorch 官方 DDP 文档明确把输入切分责任留给用户，例如使用 `DistributedSampler`；handout 的 naive DDP 同样要求把含 $n$ 个样本的全局 batch 均分给 $P$ 个设备。[1][5]

### 1.2 初始化参数广播

各 rank 可以用不同随机状态构造模型，但进入第一次 forward 前必须以 rank 0 为源，同步模型初值。`torch.distributed.broadcast(tensor, src=0)` 把源 rank 的 tensor 广播给整个 group；所有参与进程上的 tensor 必须有相同元素数，非源 rank 的 tensor 用于接收结果。[2]

对模型参数逐个广播后，有：

$$\theta_0^{(0)}=\theta_0^{(1)}=\cdots=\theta_0^{(P-1)}=\theta_0$$

handout 的算法明确要求从 rank 0 广播模型参数。PyTorch 官方 DDP 的 `init_sync=True` 也会在构造期校验参数形状并广播参数和 buffer；初始化完成后，正式 DDP 不会在每个 optimizer step 后再次广播参数，而是假设各 rank 用同一梯度执行同一更新。[1][5]

初始化同步有三个边界：

1. **冻结参数也属于模型状态。** 即使 `requires_grad=False`，它仍可能参与 forward，因此一般也必须广播。
2. **buffer 不是 parameter。** 只广播 `module.parameters()` 不会同步 BatchNorm running statistics 等 buffer。官方 DDP 会在初始化时同步 buffer，并可在 forward 期继续同步；只实现 handout 最小参数广播的 wrapper 不自动具备这一语义。[5][6]
3. **已有 optimizer state 不会随参数广播自动同步。** 最简单的做法是在参数广播后，以相同参数顺序和超参数分别构造 optimizer，使初始 state 相同；若从 checkpoint 恢复，则必须让各 rank 加载同一 optimizer state。[1][10]

### 1.3 本地 forward 与 backward

rank $r$ 在相同参数 $\theta_t$ 上处理本地数据分片，计算本地 loss 并调用 `backward()`。PyTorch 的 `Tensor.backward()` 把梯度累积到叶子 tensor 的 `.grad`，不会自动覆盖已有梯度，因此每个训练更新前需要正确清空梯度。[8]

在 all-reduce 之前，各 rank 的梯度通常不同：

$$g_t^{(r)}=\nabla_{\theta}L_t^{(r)}(\theta_t)$$

本地 forward/backward 可以在不同 rank 上并行执行，但 collective 必须由所有参与 rank 以兼容的顺序和 tensor 元数据共同进入。官方 DDP 把构造、forward 和输出求导列为分布式同步点，并要求各进程具有一致的参数注册顺序。[5]

### 1.4 逐参数梯度 all-reduce 与除以 world size

`torch.distributed.all_reduce` 对所有参与进程的 tensor 做归约，使每个进程得到最终结果；该操作原地修改输入，默认 reduction 是 `SUM`。[3]

因此，对每个满足 `parameter.grad is not None` 的唯一参数张量执行：

$$g_t\leftarrow\sum_{r=0}^{P-1}g_t^{(r)}$$

随后必须除以 world size：

$$\bar g_t=\frac{1}{P}\sum_{r=0}^{P-1}g_t^{(r)}$$

`SUM all_reduce` 本身不会替 naive DDP 完成这个除法。可以在 all-reduce 前把本地梯度除以 $P$，也可以在 all-reduce 完成后除以 $P$；在实数数学中二者等价。为了让语义直观，本报告后续都按“先 SUM，再除以 $P$”描述。[1][3]

每个 rank 必须对同一组参数以同一顺序发起 collective。若某个参数只在部分 rank 上得到 `.grad`，简单地按本地 `grad is not None` 分支会让 collective 序列不一致，可能挂起。naive 实现应限定所有 rank 使用相同计算图，或为未使用参数设计一致的同步协议；官方 DDP 用 `find_unused_parameters` 等机制处理更一般的图。[5]

### 1.5 本地 optimizer step

梯度同步完成后，每个 rank 在本地调用同一 optimizer 的 `step()`。PyTorch 的分布式说明明确采用“每个进程维护自己的 optimizer 并执行完整更新”的方式；因为梯度已经跨进程平均且相同，不需要在每一步之后重新广播参数。[4][5]

完整迭代的依赖关系是：

```text
zero_grad
-> local forward
-> local backward
-> for each unique parameter gradient: SUM all-reduce
-> divide each synchronized gradient by world size
-> optimizer.step
```

optimizer step 不能早于梯度通信完成。同步 collective 返回后可直接继续；若改为 `async_op=True`，则必须保存 work handle，并在 optimizer 读取梯度前按官方异步语义等待完成。[1][3]

## 2. 为什么它等价于全局 batch 的 mean loss

### 2.1 核心：先求本地 mean，再对 rank 求平均

把全局 batch 均分给 $P$ 个 rank，每个 rank 处理 $B$ 个样本，并统一使用 mean loss。rank $r$ 的本地 loss 为：

$$L^{(r)}(\theta)=\frac{1}{B}\sum_{i=1}^{B}\ell(\theta;z_{r,i})$$

对应的本地梯度为：

$$g^{(r)}=\nabla_{\theta}L^{(r)}(\theta)=\frac{1}{B}\sum_{i=1}^{B}\nabla_{\theta}\ell(\theta;z_{r,i})$$

DDP 再对 $P$ 个本地梯度求平均：

$$\bar g=\frac{1}{P}\sum_{r=0}^{P-1}g^{(r)}=\frac{1}{PB}\sum_{r=0}^{P-1}\sum_{i=1}^{B}\nabla_{\theta}\ell(\theta;z_{r,i})$$

右侧正是对全部 $PB$ 个样本求 mean loss 后得到的梯度。因此本质只有一句话：

> 每个 rank 对等大的本地 batch 使用 mean loss，再对各 rank 的梯度求平均，就等价于在全局 batch 上使用 mean loss。

### 2.2 本实验中的落地

作业测试使用默认 `reduction="mean"` 的 `MSELoss`，两个 rank 的本地 batch 和输出形状相同，因此直接满足上述条件。[9]

训练代码只需始终遵循同一个约定：

1. 全局 batch 均匀切分到各 rank；
2. 每个 rank 使用 mean loss；
3. all-reduce 求梯度和后除以 world size。

这样即可得到全局 batch 的平均梯度，不必再引入 sum loss 等额外分支。[1][5]

## 3. 为什么各 rank 会持续同步

### 3.1 更新不变量

把 optimizer 的一次更新抽象为确定性状态转移：

$$\left(\theta_{t+1},s_{t+1}\right)=F\left(\theta_t,s_t,\bar g_t,h_t\right)$$

其中 $h_t$ 表示该步使用的 optimizer 超参数，例如 learning rate、weight decay 和 momentum 系数。若对任意 rank $r$ 都有：

$$\theta_t^{(r)}=\theta_t,\qquad s_t^{(r)}=s_t,\qquad \bar g_t^{(r)}=\bar g_t,\qquad h_t^{(r)}=h_t$$

那么各 rank 向相同函数 $F$ 输入相同值，输出也相同：

$$\theta_{t+1}^{(r)}=\theta_{t+1},\qquad s_{t+1}^{(r)}=s_{t+1}$$

初始化参数广播建立 $t=0$ 的参数基例；相同 optimizer 初态建立 state 基例；每轮梯度平均和相同 optimizer step 保持归纳不变量。handout 明确以这一理由说明参数和 optimizer state 会保持同步，PyTorch 官方 DDP 也依赖同一假设而不在每步更新后广播参数。[1][5]

### 3.2 optimizer state 的实际含义

optimizer state 不只包括张量，也包括参数组元数据。PyTorch `Optimizer.state_dict()` 的 `state` 按参数保存当前状态，`param_groups` 保存 learning rate、weight decay 等元数据以及参数 ID 顺序。[10]

因此“相同 optimizer state”至少要求：

1. optimizer 类型和所有参数组超参数相同；
2. 参数在各 optimizer 参数组中的顺序和归属相同；
3. momentum、Adam moments、step counter 等状态相同；
4. scheduler 或其他外部逻辑在各 rank 上以相同步数修改超参数；
5. 所有 rank 都在同一更新边界执行或跳过 optimizer step。

只广播模型参数并不能修复已经分叉的 optimizer state。若某个 rank 使用不同 learning rate、跳过一次 step，或加载了不同 checkpoint，后续即使梯度再次相同也不能保证恢复同步。[5][10]

## 4. 冻结参数、未使用参数与 tied weights

### 4.1 冻结参数

`requires_grad=False` 用于把叶子 tensor 排除出梯度累积。PyTorch autograd 文档说明，backward 只会把梯度累积到 `requires_grad=True` 的叶子 tensor；冻结参数不会更新 `.grad`。[7]

naive DDP 对冻结参数应采用：

1. **初始化时广播。** 冻结只表示不求参数梯度，不表示各 rank 可以持有不同参数值。
2. **训练时不 all-reduce。** 没有 `.grad` 的冻结参数不需要梯度通信。
3. **optimizer 不更新。** 通常 optimizer 会忽略没有梯度的参数；更清晰的做法是只把可训练参数交给 optimizer。

“是否广播”由它是否属于需要一致的模型状态决定；“是否 all-reduce”由该步是否存在梯度决定。这两个条件不能混为一谈。[2][7]

### 4.2 动态未使用参数

一个 `requires_grad=True` 参数也可能因为控制流或 loss 路径在某一步未被使用，从而得到 `grad is None`。如果所有 rank 对同一参数都没有梯度，全部跳过是安全的；如果只有部分 rank 跳过，则逐本地 `.grad` 判断会破坏 collective 调用一致性。官方 DDP 的 `find_unused_parameters` 正是为更一般的未使用参数图提供显式处理。[5]

最小 naive 方案可以把“所有 rank 使用相同计算图，产生相同的 grad-present 集合”写成前置条件，并在实验中验证每一步各 rank 的梯度存在性模式一致。

### 4.3 Tied weights

tied weights 指多个模块属性引用同一个 `nn.Parameter` 对象。PyTorch `Module.named_parameters()` 的 `remove_duplicate` 默认为 `True`，会从迭代结果中去除重复参数；`parameters()` 同样基于这一参数遍历机制。[6]

同一个 Parameter 在 forward 中被使用多次时，autograd 会把各条路径的贡献累积到同一个叶子 `.grad`；`backward()` 对叶子梯度采用累积语义。[8]

因此正确顺序是：

1. 用默认去重的 `module.parameters()` 或 `named_parameters(remove_duplicate=True)` 获取唯一 Parameter。
2. 初始化时只广播该 Parameter 一次。
3. backward 让所有使用路径先汇入同一个 `.grad`。
4. 对这个合并后的 `.grad` 只 all-reduce 一次。
5. optimizer 只接收该 Parameter 一次。

若按模块路径手工遍历并关闭去重，同一 storage 可能被重复广播、重复 all-reduce 或重复加入 optimizer。尤其是重复 all-reduce 会把已经包含全部 tied-use 局部贡献的梯度再次归约，破坏梯度尺度。[3][6][8]

## 5. Gloo、NCCL 与 CPU/GPU 边界

### 5.1 官方支持矩阵

PyTorch 官方 backend 表对 `broadcast` 和 `all_reduce` 给出的边界是：

| Backend | CPU tensor | CUDA tensor | 推荐用途 |
|---|---:|---:|---|
| Gloo | 支持 | 支持这些 collective | CPU 分布式训练；GPU 问题排查时可作较慢 fallback |
| NCCL | 不支持 | 支持 | CUDA GPU 分布式训练 |

官方经验规则是 CUDA GPU 训练使用 NCCL，CPU 训练使用 Gloo；NCCL 是 GPU 训练的推荐 backend，Gloo 在 GPU 上通常更慢。[4][5]

所以不能用 NCCL process group 直接 all-reduce CPU 梯度。若模型和梯度在 CUDA 上，naive DDP 应让 collective 直接作用于 CUDA `.grad`；为了改用 Gloo 而把每个梯度显式复制到 CPU、通信后再复制回 GPU，会增加设备间传输，并且测到的是另一条数据路径，不应和直接 NCCL 结果混为一谈。[3][4]

### 5.2 一进程一设备

单机 $P$ 张 CUDA GPU 的常规配置是 $P$ 个进程，每个进程设置自己的当前设备，并确保该进程独占 NCCL 所使用的 GPU。PyTorch 官方 DDP 文档给出 `torch.cuda.set_device(i)` 或等价的 accelerator API；`init_process_group` 文档警告多个 NCCL 进程共享 GPU 可能导致 deadlock 或 invalid usage。[4][5]

模型参数、输入和 loss 计算必须位于该 rank 对应的 device 上。collective 的对应 tensor 还必须在各 rank 间具有兼容的元素数、dtype 和布局；broadcast 文档至少明确要求所有参与 tensor 元素数相同，DDP 进一步要求各进程参数注册顺序和 stride 一致。[2][5]

### 5.3 同步与计时边界

`async_op=False` 表示 Python 调用不返回异步 work handle，但 CUDA 工作本身仍具有异步执行语义。handout 要求 GPU benchmark 使用 `torch.cuda.synchronize()`，并提醒即使同步 collective 调用返回，也可能只是工作已排入 GPU；若要测量通信完成时间，计时边界必须等待 GPU 完成。[1]

因此后续实验必须明确区分：

1. host 发起 collective 的 API 时间；
2. collective 结果可以被后续 GPU 工作安全依赖的时间；
3. 主机显式等待 GPU 完成后的端到端通信时间。

这些计时要求属于紧随本题之后的独立问题 `naive_ddp_benchmarking`。本报告只完成 `naive_ddp` 的 CPU/Gloo 正确性实验，不把 pytest 的墙钟时间解释成 GPU 通信性能。

## 6. 通信复杂度与可并行性

### 6.1 调用次数与通信量

naive DDP 每次更新发起 $K$ 次 all-reduce，总梯度 payload 为：

$$M=\sum_{j=1}^{K}m_j$$

handout 的理想 ring all-reduce 由 ring reduce-scatter 和 ring all-gather 组成。对一个大小为 $S$ 的 tensor，每个 rank 的理想发送量对应 $2\frac{P-1}{P}S$，忽略启动延迟时耗为 $2\frac{P-1}{P}\frac{S}{C}$，其中 $C$ 是每 rank egress bandwidth。[1]

把同一模型的所有梯度代入，纯 payload 项为：

$$V_{\mathrm{rank}}=2\frac{P-1}{P}M$$

从渐近量级看，在固定 $P$ 时每 rank 通信量是 $O(M)$；但调用次数仍是 $K$。将梯度 flatten 成一个 tensor 不改变 $M$ 的量级，却能把 collective 次数从 $K$ 降为 1。handout 明确指出每次通信调用都有开销，所以许多小参数会让 naive 方案受逐调用固定成本影响。[1]

可以用含启动成本的分析模型表达这一点：

$$T_{\mathrm{comm,naive}}\approx\sum_{j=1}^{K}\left(\alpha_P+\beta_Pm_j\right)=K\alpha_P+\beta_PM$$

这里 $\alpha_P$ 汇总一次给定 world size collective 的固定调度和协议开销，$\beta_P$ 汇总单位字节成本。该式是解释趋势的模型，不声称 Gloo 或 NCCL 在所有消息大小、拓扑和版本下都采用同一具体算法。

### 6.2 哪些工作可以并行

可并行部分：

1. 各 rank 的本地 forward 彼此并行。
2. 各 rank 的本地 backward 彼此并行。
3. collective 内部的数据传输和归约可由 backend 在设备与链路间并行执行。[1][4]

naive 调度中不能被隐藏的部分：

1. 梯度通信在整个 backward 结束后才开始，因此不能与该 backward 重叠。
2. 若 Python 循环逐个执行同步 all-reduce，参数间 collective 位于同一串行关键路径。
3. optimizer step 必须等待全部梯度同步完成。

于是 naive step 的关键路径近似为：

$$T_{\mathrm{step,naive}}\approx T_{\mathrm{forward}}+T_{\mathrm{backward}}+T_{\mathrm{comm}}+T_{\mathrm{optimizer}}$$

handout 的改进版本在参数梯度 ready 时发起异步 all-reduce，使部分通信与后续 backward 计算重叠；正式 DDP 则把参数放入 bucket，使 bucket 归约有机会和 backward 重叠。[1][5]

### 6.3 慢 rank 与负载均衡

collective 要等待所有参与者进入匹配操作，所以一步训练受最慢 rank 影响。handout 要求全局 batch 能被设备数整除，以避免某些 rank 处理更多样本并成为瓶颈。[1]

即使样本数相同，变长序列、条件计算、数据加载抖动或设备差异也可能让 rank 到达 collective 的时间不同。实验应同时保证数据量可比，并观察逐 rank 时间，而不能只记录 rank 0 的局部计算时间。

## 7. 与梯度累积的关系

### 7.1 单设备梯度累积基础

`backward()` 会把新梯度加到叶子已有 `.grad` 中，因此可以连续处理 $A$ 个 microbatch，只在累积窗口开始前清梯度，并在窗口结束后执行一次 optimizer step。[8]

每个 microbatch 都使用 mean loss。为了让 $A$ 个 microbatch 的累积结果等价于逻辑大 batch 的 mean loss，每次 backward 使用 $L_a/A$：

$$\nabla_{\theta}L_{\mathrm{window}}=\sum_{a=1}^{A}\nabla_{\theta}\frac{L_a}{A}$$

这和 world-size 平均是两个独立归一化维度：$1/A$ 负责 microbatch 维，$1/P$ 负责 rank 维。

### 7.2 每个 microbatch 都同步

若 naive DDP 在每次 microbatch backward 后都 all-reduce 当前 `.grad` 并除以 $P$，数学上仍可保持正确。设前 $a-1$ 次已经得到各 rank 相同的累积梯度 $G_{a-1}$，第 $a$ 次加入本地贡献 $g_a^{(r)}$，则平均后：

$$\frac{1}{P}\sum_{r=0}^{P-1}\left(G_{a-1}+g_a^{(r)}\right)=G_{a-1}+\frac{1}{P}\sum_{r=0}^{P-1}g_a^{(r)}$$

相同的旧梯度不会被额外放大，因为它在每个 rank 上相同，跨 rank 平均后仍是自身。但这种方式每个 optimizer update 执行 $A K$ 次 collective，通信开销通常没有必要。[3][8]

### 7.3 只在累积边界同步

更直接的方案是前 $A-1$ 个 microbatch 只做本地累积，在第 $A$ 个 microbatch backward 完成后，对完整窗口梯度逐参数 all-reduce 一次。这样每个 optimizer update 仍只有 $K$ 次 collective。

由于每次 backward 已使用 $L_a/A$，窗口结束时只需再对 rank 维求平均：

$$\bar g_{\mathrm{window}}=\frac{1}{PA}\sum_{r=0}^{P-1}\sum_{a=1}^{A}g_a^{(r)}$$

正式 PyTorch DDP 提供 `no_sync()` context 来推迟梯度同步，并要求 forward 也放在 context 内；自定义 naive wrapper 若只在显式 `ddp_on_after_backward` 中通信，则可以直接只在累积边界调用该函数。[5]

## 8. Naive 方案的局限

### 8.1 性能局限

1. **collective 数量多。** 每个参数一次 all-reduce，小参数多时固定调用成本显著。[1]
2. **没有 compute/communication overlap。** 必须先完成整个 backward，再开始通信。[1]
3. **同步关键路径长。** optimizer 必须等待 $K$ 次梯度同步全部结束。
4. **小 tensor 难以利用链路带宽。** flatten 或 bucket 能提高单次消息大小，但 naive 方案没有这一层聚合。[1][5]

### 8.2 内存局限

每个 rank 都保留完整参数、完整梯度和完整 optimizer state。数据并行只切分输入 batch，不切分模型训练状态；PyTorch 也单独提供 optimizer state sharding 方案来减少这种逐 rank 冗余。[1][5]

因此增加 GPU 数可以扩大有效 batch 或降低每 rank activation 负担，却不会让每 rank 的参数、梯度和 Adam moments 按 $1/P$ 缩小。

### 8.3 语义局限

1. 最小 handout 实现只要求广播参数，不等于正式 DDP 的 buffer 同步语义。[1][5]
2. 朴素 `grad is not None` 分支假设各 rank 的计算图和梯度存在性模式一致。
3. 参数注册顺序、shape、dtype 和 stride 必须跨 rank 兼容。[2][5]
4. 初始化后新增、删除或替换参数会破坏既定同步顺序；官方 DDP 同样禁止 wrap 后改变参数集合。[5]

### 8.4 它仍然有价值

naive DDP 把正确性路径暴露得最清楚：一次初始化广播、一次本地 autograd、显式梯度平均、一次本地 optimizer 更新。它适合作为 correctness reference 和后续 flatten、bucket、hook、异步通信及 FSDP 的性能基线；handout 也按这一顺序组织后续优化。[1]

## 9. 实现

### 9.1 模块接口

实现位于 [`cs336_systems/ddp.py:L15-L57`](../cs336_systems/ddp.py#L15-L57)：

| 接口 | 职责 |
|---|---|
| `_DistributedDataParallelBase.__init__` | 校验默认 process group 已初始化，保存被包装 module，记录 world size，并从 rank 0 广播初始状态 |
| `forward(*inputs, **kwargs)` | 透明调用被包装 module |
| `finish_gradient_synchronization()` | backward 后按唯一 Parameter 顺序逐个执行同步 `SUM all_reduce`，再原地除以 world size |

这个接口把 collective 细节集中在一个 module 中。训练主链路只需要：

```python
optimizer.zero_grad()
loss = loss_fn(ddp_model(local_x), local_y)
loss.backward()
ddp_model.finish_gradient_synchronization()
optimizer.step()
```

### 9.2 从单进程训练到 DDP：代码组织发生了什么变化

DDP 不只是把 `model` 换成一个 wrapper。训练程序还要从“一个进程完成全部工作”改造成“多个 rank 执行相同训练循环，但各自处理不同数据分片”。

| 训练环节 | 单进程训练 | DDP 训练 |
|---|---|---|
| 进程 | 1 个 Python 进程 | 启动 $P$ 个 worker，每个 worker 有唯一 rank |
| 通信环境 | 不需要 | 每个 worker 加入同一个 process group |
| 模型 | 只有一份模型 | 每个 rank 持有完整模型副本，构造 wrapper 时从 rank 0 广播初值 |
| 数据 | 一个进程处理整个 batch | 每个 rank 只处理互不重叠的本地 batch |
| forward/backward | 在整个 batch 上计算 | 每个 rank 独立计算本地 forward、mean loss 和 backward |
| 梯度 | 直接交给 optimizer | optimizer step 前先跨 rank 平均梯度 |
| optimizer | 一个 optimizer 更新一份模型 | 每个 rank 都有本地 optimizer；相同初值和平均梯度使更新结果保持一致 |

单进程训练循环通常是：

```python
model = Model()
optimizer = SGD(model.parameters(), lr=0.1)

for x, y in batches:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

对应的 naive DDP 组织方式是：

```python
def worker(rank: int, world_size: int):
    device = setup_process_group(rank, world_size)
    model = NaiveDistributedDataParallel(Model().to(device))
    optimizer = SGD(model.parameters(), lr=0.1)

    for global_x, global_y in batches:
        local_x = global_x.chunk(world_size)[rank].to(device)
        local_y = global_y.chunk(world_size)[rank].to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(local_x), local_y)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    cleanup_process_group()


mp.spawn(worker, args=(world_size,), nprocs=world_size)
```

这里新增了四个关键组织环节：

1. 用 `mp.spawn` 或 `torchrun` 启动多个 worker。
2. 每个 worker 初始化 process group，并根据 rank 选择设备和数据分片。
3. 用 DDP wrapper 包装本地模型副本。
4. 在 backward 和 optimizer step 之间插入梯度同步。

作业测试确实把这些环节封装在 [`tests/test_ddp.py:L27-L152`](../tests/test_ddp.py#L27-L152) 中：测试负责启动进程、建立 process group、切分数据和执行训练循环；我们的 `NaiveDistributedDataParallel` 只负责模型状态广播与梯度同步。这是一种职责分离，不代表这些训练组织步骤可以省略。

测试为了验证等价性，先让所有 rank 读取同一份完整 fixture，再通过索引手工切出本地 batch。实际训练通常不会先在每个 rank 上构造完整 global batch，而是让 `DistributedSampler` 或数据管线直接为每个 rank 提供不同的本地 batch。

#### 后续实验是否会单独涉及训练组织

handout 后续没有一道只考察“如何组织 DDP 训练程序”的独立题目。与这一环节最接近的是紧随其后的 `naive_ddp_benchmarking`：它要求用 naive DDP 训练语言模型并测量完整 step，因此必须把 worker 启动、process group、模型包装、数据生成、训练循环和结果汇总写成独立 benchmark。

本仓库对应实现位于：

1. [`scripts/benchmark_ddp.py:L132-L224`](../scripts/benchmark_ddp.py#L132-L224)：组织不同策略和重复实验的独立子进程。
2. [`cs336_systems/ddp_benchmark.py:L83-L229`](../cs336_systems/ddp_benchmark.py#L83-L229)：在 worker 内初始化 process group、构造模型和 optimizer，并执行完整训练 step。

后续 flat-gradient 和 overlap 实验复用同一套训练组织，只替换梯度通信策略。再后面的 FSDP 会改变参数、梯度和 optimizer state 的存放及通信方式，但“多 worker + process group + 本地数据分片 + 本地训练循环”的外层结构仍然相同。

### 9.3 代码位置与作用

| 代码位置 | 在执行链路中的作用 |
|---|---|
| [`tests/test_ddp.py:L27-L35`](../tests/test_ddp.py#L27-L35) | pytest 入口；分别为普通模型和 tied-weight 模型启动 2 个 worker |
| [`tests/test_ddp.py:L38-L152`](../tests/test_ddp.py#L38-L152) | worker 主流程；初始化进程组、运行全局 batch 基线和 DDP 分片训练、比较参数 |
| [`tests/common.py:L13-L94`](../tests/common.py#L13-L94) | 定义 ToyModel、tied-weight 模型、Gloo process group 和跨 rank 状态一致性检查 |
| [`tests/adapters.py:L37-L69`](../tests/adapters.py#L37-L69) | 作业测试与学生实现之间的稳定接口；负责构造 DDP wrapper，并在 backward 后触发同步 |
| [`tests/test_ddp_variants.py:L24-L67`](../tests/test_ddp_variants.py#L24-L67) | 现行三策略回归入口；直接构造 `naive`、`flat` 和 `overlap`，防止后续优化破坏 naive |
| [`cs336_systems/ddp.py:L15-L57`](../cs336_systems/ddp.py#L15-L57) | 我们实现的公共 DDP 基类和 naive 策略；执行初始状态广播、透明 forward 和逐参数梯度平均 |

`tests/` 负责组织实验和判断结果，`tests/adapters.py` 负责把作业规定的接口映射到实现，真正的 DDP 算法位于 `cs336_systems/ddp.py`。

### 9.4 测试执行链路

以当前 naive 回归命令为例：

```bash
uv run pytest tests/test_ddp_variants.py -q
```

调用链如下：

```text
pytest
└── test_all_ddp_variants_match_the_global_batch_baseline()
    └── mp.spawn(..., nprocs=2)
        └── _test_all_ddp_variants(rank, world_size)
            ├── _setup_process_group(..., backend="gloo")
            ├── 构造单进程全局 batch 基线模型
            ├── wrap_ddp(deepcopy(baseline), "naive")
            │   └── _DistributedDataParallelBase.__init__()
            │       └── 从 rank 0 广播参数和 buffer
            └── 每个训练 step
                ├── 基线：完整 global batch -> mean loss -> backward -> step
                ├── DDP：本地半批 -> mean loss -> backward
                ├── finish_gradient_synchronization()
                │   └── 每个 gradient：SUM all-reduce -> 除以 world size
                ├── DDP optimizer.step()
                └── 检查跨 rank 一致，并在 rank 0 对比全局 batch 基线
```

这条链路分别验证了：

1. wrapper 构造时是否把 rank 0 初值同步到其他 rank；
2. 不同 rank 的本地 mean gradient 是否被正确平均；
3. optimizer 读取梯度前，通信是否已经完成；
4. 更新后各 rank 参数是否相同；
5. DDP 分片训练是否与单进程 global batch 训练等价；
6. frozen parameter 和 tied weight 是否保持正确语义。

作业原始测试 [`tests/test_ddp.py`](../tests/test_ddp.py) 的主体链路相同，区别是它通过 `get_ddp()` 和 `ddp_on_after_backward()` 两个 adapter 入口调用学生实现。本题验收时 adapter 指向 naive；完成后续 overlap 题后，现行 adapter 已指向 overlap。因此，当前验证 naive 应以 `tests/test_ddp_variants.py` 中直接构造 `variant="naive"` 的路径为准。

### 9.5 初始化同步

构造函数先检查 `torch.distributed` 是否可用且默认 process group 是否已经初始化；不满足时立即抛出明确错误。随后：

```python
self._parameters_to_sync = tuple(module.parameters())
```

`module.parameters()` 返回参数迭代器；转换后，`_parameters_to_sync` 的类型是 `tuple[nn.Parameter, ...]`。例如：

```text
(
    Parameter(shape=(10, 10), requires_grad=True),
    Parameter(shape=(50,), requires_grad=False),
    Parameter(shape=(10, 50), requires_grad=True),
)
```

tuple 保存的是 Parameter 对象的引用，不会复制参数 tensor。因此 optimizer 原地更新参数后，通过 tuple 访问到的仍是更新后的对象。

这里转换为 tuple 有以下作用：

1. `module.parameters()` 是一次性迭代器；tuple 可以在初始化广播和后续每个训练 step 中反复遍历。
2. 包装时固定参数集合与遍历顺序，使所有 rank 按相同顺序提交 collective。
3. `module.parameters()` 会递归收集子模块参数，并默认去重共享的 Parameter，因此 tied weight 只出现一次。
4. tuple 本身不可增删，能够明确表达“DDP 包装后同步清单保持不变”这一约束。

该 tuple 同时保留可训练参数和冻结参数，因为二者都可能参与 forward，初始化时都需要同步。真正同步梯度时，`_dense_gradients()` 才跳过 `requires_grad=False` 或 `grad is None` 的参数。

这也带来一个明确限制：DDP 包装完成后不能新增或替换模型参数，否则 `_parameters_to_sync` 仍保存旧的参数快照，不会自动发现新的 Parameter。

保存参数快照后，初始化同步依次执行：

1. 遍历 `_parameters_to_sync`，对每个唯一 Parameter 的 detached tensor 执行 `dist.broadcast(..., src=0)`。
2. 遍历 `module.buffers()`，同样从 rank 0 广播。

#### Parameters 与 buffers 的遍历方式为何不同

parameters 会在两个阶段被反复使用：

1. wrapper 构造时广播参数初值；
2. 每个训练 step 的 backward 后收集并同步参数梯度。

因此实现把 parameters 保存为 `_parameters_to_sync`，固定参数集合和顺序，并允许后续重复遍历。

buffer 则不参与梯度计算，也不由 optimizer 更新。当前最小实现只在 wrapper 构造时广播一次 buffer，所以 `_broadcast_module_state()` 直接消费 `module.buffers()` 返回的迭代器即可，没有必要再保存一个 `_buffers_to_sync` 成员。

同步 buffer 超出了 handout 只写“参数广播”的最低要求，能够让 wrapper 构造后的初始 module state 更一致。但它仍不等价于正式 DDP 的 buffer 同步语义：例如 BatchNorm 的 `running_mean` 和 `running_var` 会在 forward 中继续更新，各 rank 后续仍可能产生不同值。

如果需要像正式 PyTorch DDP 一样在每次 forward 前同步 buffers，可以采用与 parameters 相同的快照方式：

```python
self._buffers_to_sync = tuple(module.buffers())
```

然后在每次 forward 前广播 `_buffers_to_sync`。因此当前两种遍历写法的差异来自使用生命周期不同，而不是 Parameter 和 buffer 存在必须使用不同容器的硬性限制。

广播原地修改现有 tensor，没有替换 Parameter 对象，因此不会破坏 tied-weight alias。`module.parameters()` 默认去重，同一 tied Parameter 只通信一次。

### 9.6 梯度同步

`finish_gradient_synchronization()` 的处理顺序是：

1. 跳过 `requires_grad=False` 或本轮 `grad is None` 的参数。
2. 对 sparse gradient 给出显式错误；本实现只支持 dense gradient。
3. 对 `.grad` 执行同步 `SUM all_reduce`。
4. 用 `div_(world_size)` 原地得到 rank 平均梯度。

实现没有注册 backward hook，也没有使用 `async_op=True`。因此它严格对应 `naive_ddp`，而不是后续的 `ddp_overlap_individual_parameters`。

### 9.7 Adapter 与后续题的关系

本题完成时，`tests/adapters.py` 的 `get_ddp()` 返回 `NaiveDistributedDataParallel`，`ddp_on_after_backward()` 在 backward 和 optimizer step 之间调用 `finish_gradient_synchronization()`，据此得到第 11 节的五轮结果。

仓库随后继续完成 `ddp_overlap_individual_parameters`，因此现行 [`tests/adapters.py:L37-L69`](../tests/adapters.py#L37-L69) 已按作业最终阶段切换为 `OverlappedDistributedDataParallel`。naive 类没有保留旧门面文件，而是与 flat、overlap 一起集中在 `cs336_systems/ddp.py`；[`tests/test_ddp_variants.py`](../tests/test_ddp_variants.py) 直接对三种 variant 做回归。这个演进不改变本报告记录的 naive 算法与历史实验结论。

### 9.8 已知前置条件

1. 所有 rank 必须以相同顺序构造相同的参数和 buffer。
2. 所有 rank 每一步必须产生相同的 `grad is None` 模式；rank-dependent unused parameters 不在该 naive 实现支持范围内。
3. optimizer 必须在各 rank 使用相同参数顺序、初态和超参数。
4. 只支持默认 process group 和 dense gradients。
5. 梯度累积时只应在 optimizer update 边界调用同步；否则会产生额外 collective。

## 10. 实验方法

### 10.1 环境

| 字段 | 值 |
|---|---|
| 日期 | 2026-10-01 |
| OS | Linux 5.4.143.bsk.8-amd64, x86-64 |
| CPU | Intel Xeon Platinum 8336C, 2 sockets × 28 cores，56 logical CPUs |
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu130 |
| pytest | 9.0.3 |
| CUDA device | 不可用，`torch.cuda.is_available() == False` |
| distributed backend | Gloo，可用 |
| world size | 2 |
| dtype | FP32 |
| optimizer | SGD，`lr=0.1`，无 momentum |
| loss | `MSELoss(reduction="mean")` |
| global / local batch | 20 / 10 |
| 每个 case 的更新步数 | 5 |
| 参数比较 | `torch.allclose` 默认 `rtol=1e-5, atol=1e-8` |

本实验不需要 GPU。测试文件在 CPU 环境显式选择 Gloo；`naive_ddp_benchmarking` 才是要求 1 node × 2 GPUs、NCCL 和 xl 模型的后续独立性能实验。

### 10.2 被测模型

| 模型 | 唯一 Parameter tensors | 可训练 tensors | 唯一参数元素 | 每步 FP32 梯度 payload | 每步 all-reduce |
|---|---:|---:|---:|---:|---:|
| `ToyModel` | 5 | 3 | 1,152 | 4,400 bytes | 3 |
| `ToyModelWithTiedWeights` | 4 | 4 | 1,600 | 6,400 bytes | 4 |

`ToyModel` 同时包含冻结的 linear bias 和 `no_grad_fixed_param`，所以覆盖冻结参数路径。`ToyModelWithTiedWeights` 令 `fc4.weight` 与 `fc2.weight` 引用同一 Parameter；虽然模型有五个 linear weight 属性，默认去重后只有四个唯一 Parameter。

### 10.3 测试入口与判定标准

完整调用链见第 9.4 节。naive 题完成时，通过作业原始 adapter 入口执行：

```bash
uv run pytest tests/test_ddp.py -q
```

每个模型连续训练 5 步；每一步都要求 DDP 参数与单进程 global batch 基线 `torch.allclose`，同时检查跨 rank 状态一致、冻结参数不变和 tied-weight 语义正确。

后续题已把 adapter 切换到 overlap，因此当前 naive 回归命令是：

```bash
uv run pytest tests/test_ddp_variants.py -q
```

该测试直接调用 `wrap_ddp(..., "naive")`，不受 adapter 当前指向的策略影响。

## 11. 实验结果

### 11.1 五轮稳定性结果

| 完整运行 | pytest cases | pytest 内部耗时 | 进程墙钟时间 | 结果 |
|---:|---:|---:|---:|---|
| 1 | 2 | 7.63 s | 10.50 s | 2 passed |
| 2 | 2 | 7.94 s | 10.47 s | 2 passed |
| 3 | 2 | 7.63 s | 10.16 s | 2 passed |
| 4 | 2 | 7.44 s | 10.10 s | 2 passed |
| 5 | 2 | 8.30 s | 10.81 s | 2 passed |
| 平均 ± sample std | 2 | 7.788 ± 0.338 s | 10.408 ± 0.287 s | 10/10 cases passed |

五轮合计覆盖 10 个 parameterized cases 和 50 个分布式 optimizer updates，没有失败或 hang。第 2、3、5 轮中 c10d 曾输出 `localhost:12390` 的非致命 socket 重试警告，但 process group 随后正常建立，所有断言通过；这是连续复用固定 rendezvous 地址时的启动日志，不是梯度同步失败。

### 11.2 正确性结论

| 检查项 | 证据 | 结果 |
|---|---|---|
| rank 0 初始化状态成为统一状态 | 不同 seed 构造模型；wrapper 后逐参数检查并 all-gather `state_dict` | 通过 |
| 分片训练等价于全局 batch | 每 rank 10 个互斥样本，对比单进程 20 样本基线 | 5 步均通过 |
| 冻结参数不参与更新 | `ToyModel` 的 frozen bias 与 `no_grad_fixed_param` 每步保持不变 | 通过 |
| tied weights | tied-weight 模型完成相同的初始化与五步基线比较；实现不替换 Parameter | 通过 |
| 重复运行稳定性 | 完整 pytest 独立执行 5 次 | 5/5 通过 |

测试没有直接保存并逐张量比较 all-reduce 后的梯度；它通过每一步 optimizer 后与全局 batch 基线的参数相等，端到端验证“广播、平均梯度、step 顺序”的组合行为。这里的结论是 `torch.allclose` 数值一致，不是 bitwise equality。

### 11.3 如何解释耗时

上述秒数包含 Python/pytest 启动、两次 parameterized case、每次两进程 spawn、process-group rendezvous、数据加载、基线计算和 teardown，不能视为单步 DDP 延迟，也不能用于推断 NCCL 性能或通信占比。本题验证的是正确性。

从实现结构仍可确定：`ToyModel` 每步有 3 次同步 all-reduce，tied-weight 模型每步有 4 次；这些调用全部位于 backward 之后且串行执行。若要获得性能结论，必须在有两张 GPU 的环境按 `naive_ddp_benchmarking` 单独测量，并在 CUDA 计时边界显式同步。

## 12. 结论

1. 实现完成了 rank 0 初始参数和 buffer 广播、透明 forward，以及 backward 后逐唯一参数执行的同步 `SUM all_reduce` 和 world-size 平均。
2. CPU/Gloo 目标测试连续五轮全部通过，覆盖不同 rank 初值、互斥数据分片、五步参数等价、冻结参数和 tied weights。
3. 当全局 batch 被均匀切分、每个 rank 使用 mean loss，并对梯度做 world-size 平均时，DDP 更新等价于单进程在全局 batch 上的更新。
4. 该实现的每步通信调用数是 $K$，总梯度 payload 是 $M$，且通信完全处于 backward 后的关键路径；flatten、bucket 和 asynchronous overlap 属于后续优化问题。
5. 当前环境无 CUDA device，但不影响本题的 CPU/Gloo 正确性验收。双 GPU/NCCL 的 xl 模型性能测量属于 `naive_ddp_benchmarking`，本报告没有虚构该实验结果。

## 13. 关键一手来源

1. [Stanford CS336 Spring 2026 Assignment 2 handout](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)
2. [PyTorch `torch.distributed.broadcast`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)
3. [PyTorch `torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
4. [PyTorch distributed backends and backend selection](https://docs.pytorch.org/docs/stable/distributed.html#backends)
5. [PyTorch `DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
6. [PyTorch `Module.named_parameters`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)
7. [PyTorch Autograd mechanics: setting `requires_grad`](https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad)
8. [PyTorch `Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)
9. [PyTorch `MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
10. [PyTorch `Optimizer.state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html)
