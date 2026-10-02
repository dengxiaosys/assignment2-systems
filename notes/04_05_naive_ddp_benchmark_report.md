# Naive DDP Benchmark 详细实验报告

## 1. 问题定义

### 1.1 要回答的问题

本报告只研究 Stanford CS336 Assignment 2 的 `naive_ddp_benchmarking`：使用前一题实现的 naive DDP 训练语言模型，在单机 2 GPU 上测量：

1. 一个完整训练 step 的墙钟时间；
2. 其中暴露在关键路径上的梯度同步时间；
3. 梯度同步时间占完整 step 时间的比例。

handout 要求 naive DDP 在 backward 完成后，对每个参数的梯度分别执行 all-reduce；正式配置为 1 node × 2 GPUs 和 Section 2.1.2 的 `xl` 模型。[1] 本题不是单独的 all-reduce microbenchmark，也不研究 flatten、bucket、异步 hook、通信计算重叠、FSDP 或多机扩展。

### 1.2 固定实验配置

| 项目 | 本报告采用的口径 |
|---|---|
| 节点 / rank / GPU | 1 / 2 / 2，每个进程独占一张 GPU |
| backend | NCCL |
| 模型 | `xl`：$d_{\text{model}}=2560$，$d_{\text{ff}}=10240$，32 层，32 个 attention heads |
| 词表大小 | 10,000 |
| context length | 512；handout 规定未特别说明时使用 512 |
| batch size | 全局 batch 4，每个 rank 的本地 batch 为 2 |
| 数值设置 | FP32、eager mode；不启用 autocast 或 `torch.compile` |
| 训练工作 | forward、平均 cross-entropy、backward、逐参数梯度同步、AdamW step |
| warmup / 正式采样 | 5 个完整 warmup steps / 10 个完整 measurement steps |
| 主要统计口径 | 每轮先对 rank 取最大值，再对 10 轮求均值和标准差 |

batch 口径来自两处 handout 约束：模型表的默认 batch size 是 4，而 naive DDP 算法把一个含 $n$ 个样本的全局 batch 均分给 $P$ 个设备。[1] 因而这里固定 $P=2$、$B_{\text{global}}=4$、$B_{\text{local}}=2$，不能把每卡 batch 4 得到的全局 batch 8 与本结果混为一谈。

### 1.3 完成状态

- DDP 与 benchmark 实现：已完成。
- 2-rank CPU/Gloo 正确性与三次重复工程 smoke：已完成。
- 1 node × 2 GPUs、NCCL、`xl` 的正式实验：硬件前置条件不满足，runner 已 fail-fast；本文不伪造数值。

## 2. 被测对象与计时语义

### 2.1 一次训练 step 的边界

主结果把一次 step 定义为以下有序区间：

1. 本地 forward；
2. cross-entropy loss；
3. 本地 backward；
4. naive DDP 逐参数梯度同步；
5. 本地 AdamW `optimizer.step()`。

`optimizer.zero_grad(set_to_none=True)` 在计时起点之前执行，以匹配仓库已有端到端 benchmark 的口径；输入和 targets 也在计时前生成并常驻各 rank 的 GPU。进程创建、process-group 初始化、模型构造、初始参数广播、数据生成与搬运、warmup、rank 间汇总、正确性检查和销毁进程组均不计入 step。

该边界测的是稳定态训练 kernel 与通信的端到端延迟，不是启动一个训练作业的总耗时。若后续决定把 `zero_grad` 或数据加载计入生产训练 step，必须作为不同口径另报，不能直接覆盖本表。

### 2.2 CUDA 异步执行为什么会让朴素计时失真

CUDA 操作默认异步：Python 调用通常只把工作排入 GPU stream，函数返回并不表示 GPU 已执行完。PyTorch 官方因此要求在墙钟计时边界同步设备，或使用 CUDA events。[3][4]

NCCL 也遵循 CUDA stream 语义。NVIDIA 文档明确说明，collective API 在操作成功 enqueue 到给定 stream 后即可返回，collective 随后在 GPU 上异步执行。[5] PyTorch 2.11 的 `ProcessGroupNCCL` 源码还显示，通信运行在 NCCL stream 上，并用 event 让该 stream 等待当前计算 stream 中生产输入 Tensor 的工作。[6]

因此，以下值不是梯度通信耗时：

- 只在 `dist.all_reduce(...)` 前后读取 CPU 时钟；
- 只因为 `async_op=False` 就假定 GPU collective 已完成；
- 只测 Python 的逐参数循环开销。

PyTorch 的 `all_reduce` Python 入口在 `async_op=False` 时调用 `work.wait()` 后返回 `None`，但 CUDA 是否已经执行完成仍受上述 stream 语义约束。[12] 本报告因此采用主机墙钟加显式设备同步。每个计时区间开始前清空此前 GPU 工作，区间结束后调用 `torch.cuda.synchronize(device)`；该 API 会等待指定设备所有 stream 中的 kernel 完成，因此也覆盖 NCCL stream。[4]

### 2.3 总 step 与梯度同步的具体计时边界

每个 rank 的单轮逻辑边界如下：

```text
zero_grad()                         # 不计时
dist.barrier()
torch.cuda.synchronize(device)      # barrier 和此前工作不计时

step_start
forward -> loss -> backward

# naive
torch.cuda.synchronize(device)      # backward GPU 工作不计入同步阶段

sync_start
finish_gradient_synchronization()
torch.cuda.synchronize(device)
sync_end

optimizer.step()
torch.cuda.synchronize(device)
step_end
```

设 rank $r$ 在第 $j$ 轮的本地总时间和同步阶段时间分别为 $T^{(r)}_{\text{step},j}$ 与 $T^{(r)}_{\text{sync},j}$。计时器使用 `timeit.default_timer()` 或等价的单调高分辨率时钟。

这里的 $T_{\text{sync}}$ 是**调用方可见的梯度同步阶段时间**，包括：

- 较早到达 collective 的 rank 等待较晚 rank；
- 逐参数 Python 调用和 NCCL launch 开销；
- NCCL collective 的设备执行；
- SUM 后除以 world size 的本地 GPU kernel；
- 结束处的设备同步开销。

它不是纯网络链路时间，也不是 Nsight 中 NCCL kernel duration 的简单和。若要进一步拆分，可把 Nsight Systems 的 NCCL kernel 总时长作为辅助指标，但 handout 主结果仍应使用端到端可见时间。

### 2.4 总时间与通信占比

naive DDP 不把通信与 backward 重叠，所以概念上：

$$T_{\text{step}}\approx T_{\text{forward}}+T_{\text{loss}}+T_{\text{backward}}+T_{\text{sync}}+T_{\text{optimizer}}$$

主报告的通信占比定义为 $\rho_{\text{sync}}=100\%\times\overline{T_{\text{sync,max}}}/\overline{T_{\text{step,max}}}$。这里使用“两个均值之比”，而不是先计算每轮比例再平均。报告必须写清公式，避免不同聚合顺序产生看似矛盾的百分比。

## 3. 分布式同步与 rank-max 聚合

### 3.1 为什么不能只报告 rank 0

同步训练只有所有 rank 都完成本轮，下一轮才能稳定推进。某个 rank 的独立时长可能较短，但实际迭代吞吐受最慢 rank 限制。handout 也明确指出不同 rank 的时间可能不同，应跨 rank 汇总。[1]

每个 measurement step 开始前执行 barrier，并在 barrier 后同步本地 GPU，再启动计时。barrier 本身不计入 step；它的用途是给同一轮样本建立共同起点，避免把上一轮的漂移带入下一轮。

### 3.2 逐轮 rank-max

对同一轮 $j$，先计算 $T_{\text{step,max},j}=\max_{0\le r<P}T^{(r)}_{\text{step},j}$ 和 $T_{\text{sync,max},j}=\max_{0\le r<P}T^{(r)}_{\text{sync},j}$。然后仅对这 10 个逐轮最大值计算 mean、population standard deviation、median、p95、min 和 max。主表至少报告 mean ± std；其余统计用于识别抖动与离群值。

注意，两个最大值可能来自不同 rank，因此 $\overline{T_{\text{sync,max}}}/\overline{T_{\text{step,max}}}$ 是“最慢阶段相对最慢 step”的保守聚合指标，不是某个固定 rank 的严格时间分解。原始数据仍应保留每个 rank、每一轮的成对样本，便于补充检查。

### 3.3 汇总操作不能污染计时

每个 rank 先把 10 轮本地结果保存在 Python 列表中；全部 measurement steps 完成后，再一次性使用 `dist.all_gather_object` 或 `dist.gather_object` 收集小型元数据。汇总 collective、JSON 写盘和打印都在计时区间外。

如果每轮都在计时区间内 gather 时间，会测到“训练 + 结果采集”，不再是题目要求的训练 step。只打印 rank 0 的本地值则无法观察另一个 rank 是否更慢。

## 4. NCCL 与 GPU 实验约束

### 4.1 进程和设备映射

PyTorch 官方建议 CUDA 分布式训练使用 NCCL，并要求多进程 NCCL 中每个进程独占其使用的 GPU；共享同一 GPU 可能导致 deadlock 或 NCCL invalid usage。[2] NVIDIA 同样规定一个 communicator 内不同 rank 必须映射到不同 CUDA device。[7]

正式运行应满足：

1. 使用 `torchrun --standalone --nproc-per-node=2 ...` 或等价的 `spawn`；
2. 读取 `LOCAL_RANK`，先执行 `torch.cuda.set_device(local_rank)`；
3. 以对应 device 初始化 NCCL process group；
4. 模型、输入、targets、loss 和 gradients 全部位于本 rank 的 GPU；
5. rank 0 对初始模型状态的广播在 warmup 前完成；
6. 两个 rank 以相同顺序调用 291 次对应 dtype、shape 和 count 的 all-reduce。

NCCL 官方说明 collective 必须由每个 rank 使用相同 count 和 datatype 共同调用，否则可能 hang、crash 或产生数据损坏。[8] 因此不能仅依据本地 `grad is None` 动态跳过不同参数；benchmark 应使用固定计算图，并在正式计时前验证各 rank 的 collective 清单一致。

### 4.2 需要记录的环境

| 类别 | 必须记录的字段 |
|---|---|
| GPU | 型号、数量、显存、compute capability |
| 拓扑 | `nvidia-smi topo -m`，两卡之间是 PCIe、NVLink 还是其他路径 |
| 软件 | OS、Python、PyTorch、CUDA build/runtime、driver、NCCL 版本 |
| 运行设置 | `CUDA_VISIBLE_DEVICES`、rank-to-device 映射、dtype、TF32、autocast、compile 状态 |
| NCCL | 影响传输或算法选择的显式环境变量；无覆盖也应写“默认” |
| 资源状态 | 是否独占 GPU、是否存在其他计算进程、power/clock 是否固定 |
| workload | seed、全局/本地 batch、context length、optimizer 超参数、warmup 和采样数 |

NCCL 会依据硬件拓扑自动调优，因此 GPU 型号相同并不足以保证结果可比；rank placement 和互联路径也必须记录。[2]

### 4.3 正确性与失败处理

正式采样前后至少检查：

- loss 和 gradients 均为 finite；
- 同步后抽查或完整检查对应参数梯度在两个 rank 上一致；
- 每步恰好发起预期数量的 all-reduce；
- optimizer step 后对应参数仍一致；
- 两个 rank 都产出完整的 10 轮样本。

如果 `xl`、FP32、全局 batch 4、context 512 在目标机器 OOM，应如实报告 OOM 和设备环境，不得静默缩小 batch、context 或改用 BF16 后仍标成正式设置。

## 5. Warmup 与采样协议

### 5.1 为什么 warmup 必须是完整训练 step

handout 对 NCCL benchmark 建议先运行 5 次 warmup。[1] 这里 warmup 必须执行与正式样本相同的完整路径，因为早期迭代可能包含：

- CUDA context、cuBLAS 等库和 kernel 的首次初始化；
- caching allocator 扩容；
- NCCL communicator 的建立与首次 collective 开销；
- AdamW 在第一次 `step()` 时创建一阶、二阶状态；
- shape 相关的 kernel 或编译缓存建立。

只 warm up forward 无法预热 backward、291 次 collective 和 AdamW，仍会把一次性成本混入正式样本。

### 5.2 推荐运行顺序

1. 初始化进程组、设备、随机种子、模型、naive DDP 和 optimizer。
2. 预生成各 rank 的固定输入与 targets。
3. 执行一次不计时的正确性检查。
4. 执行 5 个不记录的完整 warmup steps。
5. 同步所有 rank 和 GPU。
6. 执行 10 个正式 measurement steps，保存逐 rank 原始值。
7. 在计时后汇总、校验并写出机器可读结果。
8. 销毁 process group。

warmup 与 measurement 必须使用相同 batch shape、dtype、模型模式和同步代码路径。不要在两阶段之间调用 `empty_cache()`，也不要只保留最快的若干样本。

## 6. 通信复杂度与可并行性

### 6.1 `xl` 模型的静态通信规模

仓库 `TransformerLM` 使用独立的 token embedding 和 LM head；每层包含 2 个 RMSNorm 参数 Tensor、4 个 attention 权重 Tensor 和 3 个 SwiGLU 权重 Tensor。[9] 因而 `xl` 模型的可训练参数量为：

$$N_\theta=2Vd+L(4d^2+3dd_{\text{ff}}+2d)+d=3{,}406{,}809{,}600$$

其中 $V=10{,}000$、$d=2560$、$d_{\text{ff}}=10240$、$L=32$。参数 Tensor 数量为 $K=1+32\times9+1+1=291$。在 FP32 且所有参数都产生 dense gradient 时，总梯度 payload 为：

$$G=4N_\theta=13{,}627{,}238{,}400\ \text{bytes}\approx13.627\ \text{GB}\approx12.691\ \text{GiB}$$

这些是由配置和当前模型源码得到的静态量，不是性能实验结果。若实现改为 tied embeddings、冻结参数或改变模型结构，必须在运行时重新统计 `requires_grad=True` 且实际有梯度的 Tensor 数与字节数。

### 6.2 逐参数 all-reduce 的延迟与带宽项

设 world size 为 $P$，第 $i$ 个梯度 Tensor 有 $S_i$ bytes，$G=\sum_{i=1}^{K}S_i$。用 ring all-reduce 的简化 $\alpha$-$B$ 模型，每个 Tensor 需要 reduce-scatter 和 all-gather 两段，naive DDP 的总通信时间近似为：

$$T_{\text{comm,ring}}\approx2K(P-1)\alpha+2\frac{P-1}{P}\frac{G}{B_{\text{eff}}}$$

当 $P=2$ 时，上式化为 $T_{\text{comm,ring}}\approx2K\alpha+G/B_{\text{eff}}$。第一项揭示 naive 方案的核心问题：相同总梯度字节数被拆成 $K=291$ 次 collective，重复支付启动、调度和 rank 协调开销。第二项由总 payload 与有效互联带宽主导。NVIDIA `nccl-tests` 对 all-reduce 给出的归一化 bus-bandwidth 因子也是 $2(P-1)/P$。[10]

上述公式只用于解释趋势。NCCL 会按消息大小、拓扑和版本选择 ring、tree、协议与 channel；实际 $B_{\text{eff}}$ 也不是硬件标称带宽，所以不能用该式替代实测或据此断言底层算法。

算法量级可概括为：

- collective 调用数：每 step 为 $K$，即 $O(K)$；
- 总梯度元素数：$O(N_\theta)$；
- ring 每 rank 的单向发送量：$2(P-1)G/P$；
- ring latency rounds：$2K(P-1)$；
- 模型、梯度和 optimizer state：每个 rank 都保留完整副本，naive DDP 不做参数或状态分片。

### 6.3 哪些工作可并行，哪些不能

两个 rank 的本地 forward 和 backward 可以并行，各自只处理全局 batch 的一半。collective 内部的 GPU 传输与归约也由参与设备并行推进，但同一次 collective 必须等待所有 rank 到达并匹配调用。

本题的 naive 调度在整个 backward 结束后才进入 Python 逐参数同步循环，并在全部梯度同步完成后才执行 optimizer step。因此：

- backward 与梯度通信没有 overlap；
- 不同参数的 all-reduce 按固定顺序提交；
- optimizer 不能与尚未完成的梯度同步并行；
- 较慢 rank 的计算抖动会表现为其他 rank 在 collective 中等待；
- AdamW 仍在每个 rank 更新完整模型，其工作量不会因 batch 被切成两半而减半。

正式 PyTorch DDP 会在梯度 bucket ready 后异步发起 all-reduce，以尝试和 backward 计算重叠；本题的 naive 基线没有该能力。[11] 若把单卡 forward/backward 时间记作 $T_{\text{fb},1}$，两卡本地计算理想化为 $T_{\text{fb},1}/2$，则 $T_{\text{step},2}\gtrsim T_{\text{fb},1}/2+T_{\text{sync},2}+T_{\text{optimizer}}$。实际缩放通常还受较小本地 batch 的 GPU 利用率、291 次 collective 启动、互联带宽和 rank 不均衡影响。因而“有两张 GPU”不推出 step 延迟恰好减半；本题测量的正是这些暴露开销。

## 7. 实现与复现

### 7.1 代码路径

| 位置 | 职责 |
|---|---|
| [`cs336_systems/ddp.py:L13-L55`](../cs336_systems/ddp.py#L13-L55) | 公共状态广播、dense gradient 筛选和 naive 逐参数同步 |
| [`cs336_systems/ddp_benchmark.py:L31-L123`](../cs336_systems/ddp_benchmark.py#L31-L123) | 配置校验、完整 step、CUDA 同步边界和 NVTX range |
| [`cs336_systems/ddp_benchmark.py:L126-L326`](../cs336_systems/ddp_benchmark.py#L126-L326) | 多进程执行、rank-max 聚合、环境与原始样本写盘 |
| [`scripts/benchmark_ddp.py:L20-L224`](../scripts/benchmark_ddp.py#L20-L224) | 独立子进程、随机化重复顺序、超时清理和 suite 汇总 |

runner 对 NCCL 做 fail-fast 校验：可见 CUDA device 少于 world size 时，在创建输出目录和 worker 前抛出错误，不会静默回退 Gloo。每个 case 在独立 Python 子进程中运行；原始 JSON 保留逐 rank、逐 step 样本，suite summary 保留真实命令、随机执行顺序、返回码和 wall time。

naive DDP 在 backward 返回后先同步 device，再开始同步阶段计时，从而不把尚未完成的 backward kernel 混入 $T_{\text{sync}}$。

### 7.2 已执行的 CPU/Gloo smoke

```bash
uv run python scripts/benchmark_ddp.py \
  --backend gloo --world-size 2 --variants naive \
  --model-size xl --d-model 64 --d-ff 128 --num-layers 2 --num-heads 4 \
  --vocab-size 256 --global-batch-size 4 --context-length 32 \
  --warmup-steps 3 --measurement-steps 10 --repeats 3 --num-threads 1 \
  --output-dir benchmark_results/ddp/cpu_gloo_naive_20261002
```

这里保留 `--model-size xl` 只是记录原 preset 名；四个显式结构参数覆盖了 preset，实际是 115,008 参数的小模型。3 个 naive case 全部通过，总计得到 30 个 rank-max step 样本。原始结果位于 `benchmark_results/ddp/cpu_gloo_naive_20261002/`。

### 7.3 正式 GPU 命令

```bash
uv run python scripts/benchmark_ddp.py \
  --backend nccl --world-size 2 --variants naive --model-size xl \
  --vocab-size 10000 --global-batch-size 4 --context-length 512 \
  --warmup-steps 5 --measurement-steps 10 --repeats 3 \
  --output-dir benchmark_results/ddp/gpu_nccl_xl_naive
```

## 8. 实验结果

### 8.1 正式 GPU 实验的可行性检查

| 执行环境 | 探测结果 | 能否执行正式实验 |
|---|---|---|
| 当前开发机 | `torch.cuda.is_available() == False`，0 张 CUDA GPU，无 `nsys` | 否 |
| `cuda-via-a` | 1 张 GTX 1060 6 GB，compute capability 6.1；Nsight Systems 2024.6.2；无 PyTorch 环境 | 否，缺第 2 张 GPU，且容量不足 |
| `agent1` | 无 `nvidia-smi`，无 PyTorch | 否 |

正式命令在当前开发机退出码为 1，核心错误为 `RuntimeError: NCCL requires 2 distinct visible CUDA devices`。这是预期的前置条件拒绝，不是 benchmark case 失败。

`xl` 有 3,406,809,600 个 FP32 参数。仅参数、梯度、Adam 一阶矩和二阶矩四份静态状态就需要：

$$4\times4N_\theta=54{,}508{,}953{,}600\ \text{bytes}\approx50.765\ \text{GiB/卡}$$

该下界尚未计入 activation、临时张量、CUDA context 和 NCCL workspace。GTX 1060 每卡只有 6 GiB，所以即使凑齐两张同型号 GPU 也无法运行指定 `xl` FP32 workload。正式双 GPU/NCCL 数值因此记为“硬件前置条件不满足”，不能填 0 或用缩小模型替代。

### 8.2 CPU/Gloo 工程 smoke 结果

| 字段 | 值 |
|---|---|
| 日期/代码基线 | 2026-10-02；base commit `31cc0ce` 加本次 benchmark 改动 |
| CPU/OS | Intel Xeon Platinum 8336C，56 logical CPUs；Linux 5.4.143 x86-64 |
| Python/PyTorch | Python 3.13.12；PyTorch 2.11.0+cu130，CUDA build 13.0 但 runtime device 不可用 |
| backend/进程 | Gloo；2 ranks；每 rank 1 intra-op thread |
| 模型/输入 | $d=64$、$d_{\text{ff}}=128$、2 层、4 heads、$V=256$、global batch 4、context 32、FP32 |
| 训练与统计 | AdamW；3 warmup；10 samples/case；3 个独立进程级 repeats；逐 step rank max |

| repeat | rank-max step mean ± population std (ms) | rank-max sync mean ± population std (ms) | sync 占比 | step p95 (ms) |
|---:|---:|---:|---:|---:|
| 0 | 19.353 ± 1.314 | 11.957 ± 1.237 | 61.8% | 21.494 |
| 1 | 20.180 ± 0.740 | 12.835 ± 0.605 | 63.6% | 21.466 |
| 2 | 19.512 ± 0.971 | 12.000 ± 0.684 | 61.5% | 20.920 |
| 三次 repeat mean 的聚合 | 19.681 ± 0.358 | 12.264 ± 0.404 | 62.3% | 不跨 repeat 合并 p95 |

该结果只验证 runner、计时、rank-max 聚合和 naive 性能基线能够端到端工作。它表明在这个 460,032-byte 梯度、21 次 collective 的小型 CPU/Gloo workload 上，同步阶段是主要暴露开销；它不预测 291 次 NCCL collective 和 12.69 GiB 梯度的 `xl` 表现。

### 8.3 完成度结论

| 交付项 | 状态 |
|---|---|
| naive DDP benchmark runner | 完成 |
| 2-rank CPU/Gloo 三次重复 smoke | 完成，3/3 case passed |
| 逐 rank、逐 step 原始 JSON 与 suite summary | 完成 |
| 2-GPU/NCCL/`xl` 正式时间 | 受硬件阻塞，未伪造 |
| Nsight GPU trace | 本题不要求 |

## 9. 常见误测与结论边界

| 误测 | 后果 | 本报告的处理 |
|---|---|---|
| 不同步 GPU 就读主机时钟 | 主要测到 enqueue 时间 | 计时前后同步指定 GPU |
| 认为 `async_op=False` 等于 GPU 已完成 | 低估 NCCL 执行时间 | 结束边界调用 `torch.cuda.synchronize` |
| 只报告 rank 0 | 忽略最慢 rank | 每轮先取 rank max |
| 把 barrier 放进 step 区间 | 混入额外 collective | barrier 在起点之前 |
| 每轮在区间内 gather/打印 | 污染训练耗时 | 采样后一次性汇总 |
| 不做 warmup | 混入 lazy init 与 Adam state 分配 | 先做 5 个完整 steps |
| 把本地 batch 4 当成题目 batch 4 | 全局 batch 和算量翻倍 | 明确 global 4 / local 2 |
| 把同步阶段称为纯网络时间 | 忽略等待、launch 和除法 | 报告“调用方可见同步时间” |
| 把 ring 公式当作 NCCL 实际算法 | 产生无证据的机制结论 | 公式只解释量级，机制需 profiler |
| OOM 后静默改精度或尺寸 | 不再回答原题 | 原配置失败就如实报告 |

本实验只能说明指定 `xl` workload 在指定双 GPU 拓扑上的 naive DDP 稳态表现。它不能直接预测其他 GPU、更多 ranks、不同 context、混合精度、真实数据管线或正式 PyTorch DDP 的性能。尤其不能把 $T_{\text{sync}}$ 全部解释为物理链路传输时间。

## 10. 参考资料

1. [Stanford CS336 Assignment 2 handout：模型配置、benchmark 方法与 `naive_ddp_benchmarking`](./cs336_assignment2_systems_extracted.md#L81-L123)，以及[分布式 benchmark 与 naive DDP 原题](./cs336_assignment2_systems_extracted.md#L1368-L1413)。
2. [PyTorch 2.11 Distributed：backend 选择与 `init_process_group`](https://docs.pytorch.org/docs/2.11/distributed.html)。
3. [PyTorch 2.11 CUDA semantics：异步执行、准确计时与 stream](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)。
4. [PyTorch 2.11 `torch.cuda.synchronize`](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html)。
5. [NVIDIA NCCL User Guide：CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)。
6. [PyTorch 2.11 `ProcessGroupNCCL.cpp`：NCCL stream 与当前 CUDA stream 的 event 依赖](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)。
7. [NVIDIA NCCL User Guide：Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。
8. [NVIDIA NCCL User Guide：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。
9. [仓库 `TransformerLM` 源码](../../assignment1-basics/cs336_basics/model.py)。
10. [NVIDIA `nccl-tests`：all-reduce time、algorithm bandwidth 与 bus bandwidth](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)。
11. [PyTorch 2.11 DDP Design Note：bucket、autograd hook 与通信计算重叠](https://docs.pytorch.org/docs/2.11/notes/ddp.html)。
12. [PyTorch 2.11 `all_reduce` Python 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/distributed_c10d.py#L2979-L3066)。
