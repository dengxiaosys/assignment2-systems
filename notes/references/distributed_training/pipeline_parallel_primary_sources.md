# Pipeline Parallelism 一手资料研究备忘录

## 1. 结论摘要

1. GPipe 将顺序网络切成 $K$ 个连续 stage，并把大小为 $N$ 的 mini-batch 切成 $M$ 个 micro-batch；一次迭代先流水执行全部 forward，再执行全部 backward，最后只更新一次参数。这是同步的 fill-drain 调度。[GPipe §2.1-2.2](https://arxiv.org/html/1811.06965v5#S2)
2. 在各 stage 完全均衡、通信可忽略且各 micro-batch 等长时，forward 或 backward 单阶段需要 $M+K-1$ 个时隙，其中有效计算占 $M$ 个，故理想利用率为 $\eta=M/(M+K-1)$，bubble 比例为 $(K-1)/(M+K-1)$。GPipe 给出同阶结论，并报告其环境中 $M\ge 4K$ 时 bubble 可忽略；这不是跨硬件的充分条件。[GPipe §2.3](https://arxiv.org/html/1811.06965v5#S2.SS3)
3. “同步梯度等价”成立的核心不是 fill-drain 本身，而是所有 micro-batch 的 forward/backward 使用同一参数版本、按样本数正确加权梯度，并在整批结束后更新一次。它保证与同一 mini-batch 的梯度累积语义一致，不保证浮点逐 bit 相同，也不自动覆盖 BatchNorm 等跨样本算子。
4. GPipe 的 rematerialization 只长期保存 stage 边界，在 backward 时重算 stage 内 forward；其理想化峰值 activation memory 从未切分、未重算的 $O(NL)$ 降为 $O(N+(L/K)(N/M))$。[GPipe §2.3](https://arxiv.org/html/1811.06965v5#S2.SS3)
5. 1F1B 只描述 forward/backward 的交错顺序，不唯一决定优化语义。原始 PipeDream 在 steady state 交替执行 1F1B，并逐 mini-batch 更新，因此需要 weight stashing，且默认仍有跨 stage staleness；PyTorch `Schedule1F1B` 则在一次 schedule step 内累计全部 micro-batch 梯度，可保持同步更新语义。[PipeDream §3.3-3.4](https://arxiv.org/html/1806.03377#S3.SS3)；[PyTorch `Schedule1F1B`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1206-L1388)

## 2. 形式化问题模型

设网络由 $L$ 个顺序层组成，第 $l$ 层为 $x_l=f_l(x_{l-1};\theta_l)$。将连续层索引划分为 $K$ 个不重叠区间 $I_1,\ldots,I_K$，stage $k$ 的复合函数为 $F_k=\mathop{\circ}_{l\in I_k}f_l$，参数为 $\Theta_k=\bigcup_{l\in I_k}\theta_l$。GPipe 以每层估计代价 $c_l$ 构造 stage 代价 $C_k=\sum_{l\in I_k}c_l$，并尝试减小各 $C_k$ 的方差。[本地 GPipe 提取稿，§2.1-2.2](./gpipe_easy_scaling_with_micro_batch_pipeline_parallelism_extracted.md)；[原论文 §2.1](https://arxiv.org/html/1811.06965v5#S2.SS1)

把 mini-batch $\mathcal D$ 划分为 $\mathcal D_1,\ldots,\mathcal D_M$，令 $n_m=|\mathcal D_m|$、$N=\sum_m n_m$。若目标是样本平均损失，则：

$$\mathcal L(\theta;\mathcal D)=\sum_{m=1}^{M}\frac{n_m}{N}\mathcal L_m(\theta)$$

性能模型至少需要 stage 的 forward/backward 时间 $t_k^F,t_k^B$、边界 activation 字节数 $A_{k,m}$、链路带宽/时延，以及每个 stage 的参数和 activation 容量。仅用层数均分不能推出负载均衡，因为层间 FLOPs、内存和边界张量大小可能不同；GPipe 自身也明确把不均衡分区列为限制。[GPipe §2.3](https://arxiv.org/html/1811.06965v5#S2.SS3)

## 3. GPipe Fill-Drain 与 Bubble

GPipe 的一个 step 可写成：

1. 用同一组参数 $\theta^{(t)}$ 将全部 $M$ 个 micro-batch 依次注入 forward pipeline。
2. forward 全部完成后，让每个 micro-batch 的梯度沿 stage 反方向传播。
3. 累积全部 micro-batch 梯度，再产生一次 $\theta^{(t+1)}$。

PyTorch 的 `ScheduleGPipe` 实现与该结构一致：先遍历所有 forward，再遍历所有 backward，最后执行梯度归约/缩放。[PyTorch source:L1083-L1171](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1083-L1171)

在理想模型中，每个 stage 处理一个 micro-batch 的 forward 都耗时 $t_F$。第一个输出经过 $K$ 个时隙产生，此后每个时隙产生一个输出，因此 forward makespan 为 $(M+K-1)t_F$。所有设备可提供 $K(M+K-1)$ 个 stage-slot，实际计算占 $KM$ 个，故：

$$\eta_{\mathrm{ideal}}=\frac{KM}{K(M+K-1)}=\frac{M}{M+K-1},\qquad \beta_{\mathrm{bubble}}=1-\eta_{\mathrm{ideal}}=\frac{K-1}{M+K-1}$$

若 backward 在 stage 间同样均衡但耗时为另一常数 $t_B$，对两个 phase 分别应用上述推导后，总体比例不变。若 $t_k^F,t_k^B$ 不均、通信不能隐藏、stage 映射到共享设备，或 micro-batch shape 不同，该闭式公式只是不含这些开销的上界模型。

## 4. 同步梯度为何等价

当整个 step 中参数固定为 $\theta^{(t)}$ 时，梯度线性性给出：

$$\nabla_\theta\mathcal L(\theta^{(t)};\mathcal D)=\sum_{m=1}^{M}\frac{n_m}{N}\nabla_\theta\mathcal L_m(\theta^{(t)})$$

因此，等大 micro-batch 且各自 loss 为样本均值时，应将每份梯度按 $1/M$ 缩放；不等大时必须按 $n_m/N$ 缩放。PyTorch 单 stage schedule 的 `scale_grads=True` 明确对应 mean-reduced loss，并在所有 backward 完成后除以 micro-batch 数。[PyTorch source:L852-L880](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L852-L880)；[PyTorch stage source:L857-L871](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L857-L871)

该结论的边界是：

- 参数和影响 forward 的 optimizer state 在 step 内不能改变；
- 同一 micro-batch 的 forward 与 backward 必须对应同一参数版本；
- loss 必须可按样本或 token 正确分解、加权；
- 随机数、dropout、数据顺序和有状态 buffer 必须具有可比语义；
- 浮点加法次序不同会产生舍入差异，所以通常是数学等价或容差内等价，而非逐 bit 等价；
- BatchNorm 等跨 batch 计算并不天然等价。GPipe 为其单独定义了 micro-batch 训练统计与 mini-batch 移动统计策略。[GPipe §2.2、§6](https://arxiv.org/html/1811.06965v5#S2.SS2)

## 5. Activation Memory 与 Rematerialization

Fill-drain 在第一个 backward 开始前已经完成全部 forward。若保留完整 autograd graph，stage $k$ 需要同时保存 $M$ 份 stage 内 activation，近似随该 stage 的层数和整批样本数增长。

GPipe 的 rematerialization 在 forward 期间只长期保留 stage 边界输出，backward 到达 stage $k$ 时重算 $F_k$。在均匀层数和 activation 大小的简化模型中：

$$M_{\mathrm{act,peak}}=O\left(N+\frac{L}{K}\frac{N}{M}\right)$$

第一项是全部 micro-batch 的边界 activation 总量，第二项是当前正在重算的一份 micro-batch 的 stage 内 activation。它以额外 forward 计算换内存，并没有减少参数、梯度、optimizer state、通信 buffer、CUDA context 或 allocator fragmentation。[GPipe §2.3](https://arxiv.org/html/1811.06965v5#S2.SS3)

PyTorch runtime 也显式缓存每个 micro-batch 的 stage 输入/输出供 backward 使用，并在该 micro-batch backward 时弹出缓存；因此调度改变在途 micro-batch 数会直接改变 activation 生命周期。[PyTorch stage source:L949-L1008](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L949-L1008)；[L1029-L1058](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L1029-L1058)

## 6. 通信量

GPipe 只要求在相邻 stage 边界传递 forward activation 和 backward activation gradient。[GPipe §2.2-2.3](https://arxiv.org/html/1811.06965v5#S2.SS2) PyTorch source 也分别构造 activation send 与 input-gradient send P2P 操作。[PyTorch stage source:L568-L600](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L568-L600)；[L619-L680](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py#L619-L680)

令 $A_k$ 为完整 mini-batch 在边界 $k$ 的 activation 字节数，并假设其梯度 shape 和 dtype 相同，则单 step 跨所有 $K-1$ 个边界的 payload 为：

$$V_{\mathrm{step}}=2\sum_{k=1}^{K-1}A_k$$

若各边界 shape 相同，$A_k=NSdb$，其中 $S$ 为序列长度、$d$ 为 hidden size、$b$ 为每元素字节数，则 $V_{\mathrm{step}}=2(K-1)NSdb$。切成 $M$ 份不会改变总 payload，却会把消息数提高到每个方向、每个边界 $M$ 条；所以更大的 $M$ 降低 bubble，但增加启动时延、调度和小消息开销。若存在 skip connection、多输出、不同 dtype、额外 metadata 或 label 传输，应逐边界求和，不能套用 $2NSdb$。

## 7. GPipe 与 1F1B

| 维度 | GPipe fill-drain | 同步 1F1B（如 PyTorch） | 原始 PipeDream 1F1B |
|---|---|---|---|
| 顺序 | 全部 F，再全部 B | warmup 后交替 F/B，最后 cooldown | steady state 严格交替 F/B |
| 更新时机 | 整个 mini-batch 后一次 | schedule 内累积，通常 step 后一次 | 每个 mini-batch backward 后更新 |
| 参数陈旧 | 无 | 无，前提是 step 内不更新 | 有界 staleness |
| 版本缓存 | 不需要 | 不需要 | 需要 weight stashing 保证同一 micro-batch 的 F/B 使用同版本 |
| activation 峰值 | 无重算时可随 $M$ 增长 | 受 warmup 中在途数量约束，通常显著低于 fill-drain | 输入 stage 保存 NOAM 份，输出 stage 保存一份 |
| bubble | 每 step 明显 fill/drain | 仍有 warmup/cooldown；主要收益是缩短 activation 生命周期 | steady state 可无空闲，但语义不等同同步 SGD |

PyTorch `Schedule1F1B` 明确实现 warmup、1B1F steady state 与 cooldown，并在所有 backward 后才归约/缩放梯度。[PyTorch source:L1206-L1388](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py#L1206-L1388) 原始 PipeDream 则指出：weight stashing 只保证单个 stage 内 forward/backward 权重一致；若不启用 vertical sync，各 stage 仍可能使用不同时间版本，因此其默认更新不是同步 mini-batch SGD。加入 vertical sync 后才具有其论文所述的 BSP 类似语义。[PipeDream §3.4](https://arxiv.org/html/1806.03377#S3.SS4)

因此，报告中应写“同步 1F1B 降低 activation 驻留量，同时保留整批更新”，而不能笼统写“1F1B 必然引入 stale gradient”。陈旧性来自更新策略，不来自 F/B 交错本身。

## 8. 局限与有效性威胁

1. **模型可切分性**：GPipe 假设网络可表达为层序列，且单层能放入单设备；复杂 DAG、跨 stage skip connection 和单算子超显存需要额外机制。[GPipe §6](https://arxiv.org/html/1811.06965v5#S6)
2. **框架约束**：PyTorch 自动切分要求模型可被 `torch.export` 捕获；`PipelineStage` 需要已知输入输出 shape/dtype，官方文档仍将该包标为 alpha。[PyTorch Pipeline Parallelism](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)
3. **理想化性能模型**：bubble 推导假设 stage 均衡、micro-batch 同质、通信可忽略。实际吞吐由最慢 stage、链路拓扑、P2P 是否与计算重叠、kernel 对小 batch 的效率共同决定。
4. **内存公式不完整**：GPipe 的渐近式只描述 activation 主项；真实性能还受参数与 optimizer state、临时 workspace、通信 buffer 和碎片影响。rematerialization 也会增加计算量。
5. **统计语义**：micro-batch 化会改变 BatchNorm 等 batch-coupled 算子；dropout/RNG、动态控制流和不同归约顺序也会破坏严格复现。同步更新只消除 weight staleness，不消除这些差异。
6. **1F1B 外推风险**：PipeDream 的结论来自带逐 mini-batch 更新和版本管理的系统，不能直接证明同步 1F1B 的收敛或性能；PyTorch 的类名只规定调度，不规定调用方何时执行 optimizer step。
7. **实验外推风险**：GPipe 和 PipeDream 的实证来自特定 TPU/GPU、CNN/RNN/早期 Transformer 与当时网络。论文中的“低通信开销”“$M\ge4K$ 足够”等应视为测得结果，而非现代大模型集群上的普遍定律。

## 9. 一手资料

- Huang et al., *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*, arXiv:1811.06965v5, 2019：[HTML](https://arxiv.org/html/1811.06965v5)、[PDF](https://arxiv.org/pdf/1811.06965v5)；仓库内为[原始 PDF](./gpipe_easy_scaling_with_micro_batch_pipeline_parallelism.pdf)和[提取稿](./gpipe_easy_scaling_with_micro_batch_pipeline_parallelism_extracted.md)。
- PyTorch, `torch.distributed.pipelining`, 2.14：[官方文档](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)、[`schedules.py`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/schedules.py)、[`stage.py`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/pipelining/stage.py)。
- Narayanan et al., *PipeDream: Generalized Pipeline Parallelism for DNN Training*, SOSP 2019：[作者机构 PDF](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/pipedream.pdf)、[DOI](https://doi.org/10.1145/3341301.3359646)。调度与一致性细节另见原始公开稿 [arXiv:1806.03377 §3.3-3.5](https://arxiv.org/html/1806.03377#S3.SS3)。
