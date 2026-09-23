# 端到端 Benchmark：同步计时、预热与统计

## 0. 本文目标

本文对应 handout 的 `benchmarking_script`（§2.1.3），目标是：

1. 用 Assignment 1 的 `TransformerLM` 构造课程指定规模的随机模型；
2. 分别测量 forward、loss、backward 和 optimizer step；
3. 解释 CUDA 异步执行、同步边界、warmup 和统计量；
4. 给出作业 (b)、(c) 的可复现实验命令与结果记录格式；
5. 分析时间、空间复杂度和实验的可并行性。

对应代码：

- 命令入口：[benchmark.py:L1-L7](../benchmark.py#L1-L7)
- 核心实现：[cs336_systems/benchmark.py](../cs336_systems/benchmark.py)
- 回归测试：[tests/test_benchmark.py](../tests/test_benchmark.py)
- 题目原文：[cs336_assignment2_systems_extracted.md:L109-L125](./cs336_assignment2_systems_extracted.md#L109-L125)

---

## 1. 实验定义

### 1.1 固定项与自变量

handout 默认使用 `vocab_size=10000`、`batch_size=4`、`context_length=512`。脚本内置五组模型配置：

| 名称 | `d_model` | `d_ff` | `num_layers` | `num_heads` |
|---|---:|---:|---:|---:|
| `small` | 768 | 3072 | 12 | 12 |
| `medium` | 1024 | 4096 | 24 | 16 |
| `large` | 1280 | 5120 | 36 | 20 |
| `xl` | 2560 | 10240 | 32 | 32 |
| `10b` | 4608 | 12288 | 50 | 36 |

这些 preset 定义于 [cs336_systems/benchmark.py:L33-L39](../cs336_systems/benchmark.py#L33-L39)。所有维度也都能用命令行参数覆盖，便于先做低成本冒烟测试。

### 1.2 三种运行模式

| `--mode` | 执行内容 | 输出阶段 |
|---|---|---|
| `forward` | `eval()` + `torch.inference_mode()` 下只运行模型 | `forward`, `total` |
| `forward-backward` | 训练态 forward、交叉熵、backward | `forward`, `loss`, `backward`, `total` |
| `full` | 上述步骤再加 Assignment 1 的 AdamW 更新 | `forward`, `loss`, `backward`, `optimizer`, `total` |

实现见 [_run_step:L123-L179](../cs336_systems/benchmark.py#L123-L179)。

这里有两个必须固定的口径：

- `loss` 单独计时，因此 `forward` 只表示 `TransformerLM(input_ids)`。
- `model.zero_grad(set_to_none=True)` 在计时区间外执行，`optimizer` 只表示 `optimizer.step()`。

`forward` 模式不构建 autograd 图，而训练模式会构图，因此两者的 CPU 调度开销和峰值显存不同。回答作业 (b) 时，应从同一次 `--mode full` 实验读取各阶段，保证比较口径一致；`--mode forward` 更适合纯推理和后续显存实验。

### 1.3 随机输入

脚本在设备上一次性生成固定的 `input_ids` 和 `targets`，形状均为 `(batch_size, context_length)`，元素范围为 `[0, vocab_size)`，见 [L239-L240](../cs336_systems/benchmark.py#L239-L240)。数据生成不放进计时区间，否则测到的会是“随机数生成 + 模型”的混合耗时。

---

## 2. 为什么普通计时会测错 CUDA

### 2.1 CPU 调用返回不等于 GPU 已执行完

PyTorch 发起 CUDA 运算时，CPU 通常只把 kernel 排入 CUDA stream，随后立刻继续执行。若直接写：

```python
start = timeit.default_timer()
y = model(x)
elapsed = timeit.default_timer() - start
```

`elapsed` 主要测到 CPU 提交 kernel 的时间，GPU 可能仍在计算。因此脚本在每个被测阶段结束后调用 `torch.cuda.synchronize(device)`，等 GPU 队列完成后才读取结束时间，见 [synchronize 与 _timed_call:L109-L120](../cs336_systems/benchmark.py#L109-L120)。

计时边界可以理解为：

```text
前一阶段已同步 -> start -> 提交本阶段全部 kernel -> synchronize -> end
```

CPU 模式没有异步 CUDA 队列，`synchronize()` 会成为空操作，所以同一套实现也能在 CPU 上做功能验证。

### 2.2 为什么不用每个 kernel 都同步

同步粒度是“阶段”，不是“算子”。若在每个矩阵乘、softmax 后都同步，会破坏 PyTorch 正常的异步调度并显著改变程序行为。当前边界既能得到 forward/backward/optimizer 的独立墙钟时间，又不侵入模型内部。

更细粒度的 kernel 分析应交给下一节的 Nsight Systems，而不是继续往 Python 模型代码里插计时器。

---

## 3. Warmup 为什么必要

第一次执行常包含稳态迭代没有的成本：

1. CUDA context、cuBLAS/cuDNN handle 等运行时状态的延迟初始化；
2. caching allocator 首次申请并扩展显存池；
3. 某些算子的 kernel/module 首次加载与算法选择；
4. GPU 从空闲功耗状态升频；
5. `full` 模式第一次 `optimizer.step()` 为 AdamW 的一阶、二阶矩状态分配内存。

因此 warmup 必须执行与正式测量**相同的 mode、shape、dtype 和 optimizer 路径**。脚本先完整执行 `w` 步但丢弃计时，再执行 `n` 个 measurement step，见 [L242-L267](../cs336_systems/benchmark.py#L242-L267)。

`w=1` 或 `w=2` 仍可能与 `w=5` 不同，因为一次预热只保证“初始化路径至少走过一次”，不保证 GPU 时钟、缓存、内存池和库内部选择已经稳定。是否足够最终应由原始样本及标准差判断。

---

## 4. 统计口径

设某阶段的 `n` 次耗时为 $t_1,\ldots,t_n$，脚本报告总体均值与总体标准差：

$$ \bar{t}=\frac{1}{n}\sum_{i=1}^{n}t_i,\qquad \sigma=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(t_i-\bar{t})^2}. $$

实现使用 `statistics.mean` 和 `statistics.pstdev`，见 [_summarize:L182-L190](../cs336_systems/benchmark.py#L182-L190)。JSON 文件还保留每个阶段的 `samples_ms`，便于检查首个样本是否异常、计算分位数或画箱线图。

还可计算变异系数 $CV=\sigma/\bar{t}$。绝对标准差必须结合均值判断，例如 `std=1 ms` 对 `mean=2 ms` 很大，对 `mean=500 ms` 则很小。

---

## 5. 如何运行

### 5.1 先做 CPU 冒烟测试

下面只验证流程，不作为作业性能答案：

```bash
OMP_NUM_THREADS=2 uv run python benchmark.py \
  --mode full --device cpu \
  --d-model 32 --d-ff 64 --num-layers 2 --num-heads 4 \
  --vocab-size 128 --batch-size 2 --context-length 16 \
  --warmup-steps 1 --measurement-steps 2
```

### 5.2 作业 (b)：统一做 5 次预热、10 次测量

每个模型单独启动进程，减少前一个模型留下的 allocator 状态对下一个模型的影响：

```bash
mkdir -p /home/dengxiao/logs/cs336/benchmark

for size in small medium large xl 10b; do
  CUDA_VISIBLE_DEVICES=0 uv run python benchmark.py \
    --model-size "$size" \
    --mode full \
    --device cuda \
    --dtype float32 \
    --warmup-steps 5 \
    --measurement-steps 10 \
    --output-json "/home/dengxiao/logs/cs336/benchmark/${size}_w5.json"
done
```

若某个配置 OOM，应记录设备型号、显存、dtype 和失败配置，不应通过改变 batch/context 后仍把结果放入同一张对比表。也不要把不同 GPU 的耗时直接横向比较。

### 5.3 作业 (c)：只改变 warmup 次数

固定模型、mode、dtype 和 GPU，仅改变 `warmup_steps`：

```bash
for warmup in 0 1 2 5; do
  CUDA_VISIBLE_DEVICES=0 uv run python benchmark.py \
    --model-size small \
    --mode full \
    --device cuda \
    --dtype float32 \
    --warmup-steps "$warmup" \
    --measurement-steps 10 \
    --output-json "/home/dengxiao/logs/cs336/benchmark/small_w${warmup}.json"
done
```

这里每组也使用独立进程。若在同一进程中依次跑 `w=0,1,2,5`，后面的组会继承前面组已经预热过的 CUDA context 和缓存，实验变量就不再只是 `w`。

### 5.4 输出格式

终端输出每行都有字段名，例如：

```text
benchmark_config={"batch_size": 4, ...}
phase=forward mean_ms=... std_ms=... min_ms=... max_ms=...
phase=loss mean_ms=... std_ms=... min_ms=... max_ms=...
phase=backward mean_ms=... std_ms=... min_ms=... max_ms=...
phase=optimizer mean_ms=... std_ms=... min_ms=... max_ms=...
phase=total mean_ms=... std_ms=... min_ms=... max_ms=...
```

`benchmark_config` 还记录设备名、PyTorch/CUDA 版本、CPU 线程数、seed 和优化器参数，防止汇总时混入不同环境的数据。输出和 JSON 落盘逻辑见 [_print_result 与 main:L326-L359](../cs336_systems/benchmark.py#L326-L359)。

### 5.5 用 Nsight Systems 捕获正式测量区间

脚本把 measurement loop 包在 `benchmark_measurement` NVTX range 中。可让 Nsight 跳过 warmup：

```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --capture-range=nvtx \
  --nvtx-capture=benchmark_measurement \
  --output=/home/dengxiao/logs/cs336/profiles/small_full \
  uv run python benchmark.py --model-size small --mode full --device cuda
```

NVTX 标记位置见 [L253-L267](../cs336_systems/benchmark.py#L253-L267)。

---

## 6. 复杂度与显存预判

令 batch size 为 $B$、序列长度为 $S$、层数为 $L$、模型宽度为 $d$、FFN 宽度为 $f$、词表大小为 $V$。忽略逐元素算子后，forward 的主要计算量为：

$$ \Theta\!\left(LBSd^2+LBS^2d+LBSdf+BSdV\right). $$

其中 $LBS^2d$ 来自 attention score 与 attention-value 两次矩阵乘，序列变长后会成为关键项；其余主要矩阵乘对 $S$ 线性增长。Backward 通常比 forward 更慢，因为要计算输入梯度和权重梯度，但精确比例依赖 kernel、shape 和硬件，不能仅靠 FLOPs 给出固定倍数。

当前 Assignment 1 模型没有共享 embedding 与 LM head。参数量为：

$$ P=2Vd+L(4d^2+3df+2d)+d. $$

按默认 `V=10000` 计算：

| 配置 | 参数量 | 仅 FP32 参数 | FP32 参数+梯度+AdamW 两份状态的下界 |
|---|---:|---:|---:|
| small | 128.63M | 0.48 GiB | 1.92 GiB |
| medium | 423.18M | 1.58 GiB | 6.31 GiB |
| large | 969.41M | 3.61 GiB | 14.45 GiB |
| xl | 3.41B | 12.69 GiB | 50.77 GiB |
| 10b | 12.83B | 47.81 GiB | 191.22 GiB |

这还没有计入激活、临时张量和 CUDA allocator 保留空间，所以较大配置在单卡上很可能 OOM。`--dtype bfloat16` 当前表示模型参数和运算直接使用 BF16，并不等价于带 FP32 master weights/optimizer states 的标准混合精度训练，报告结果时必须明确这一点。

---

## 7. 可并行性分析

- **同一次 step 内不可随意并行**：loss 依赖 forward，backward 依赖 loss，optimizer 又依赖梯度；脚本必须按此顺序执行。
- **同一 GPU 上不要并发跑多组 benchmark**：多个进程会争抢 SM、显存带宽和 allocator，所得耗时不再代表单任务性能。
- **不同配置可以跨 GPU 并行**：不同模型规模或 warmup 组之间没有数据依赖，可在独占的不同 GPU 上同时运行。
- **重复样本不建议拆到不同 GPU 后混合统计**：GPU 型号、温度、时钟和系统负载可能不同。最好在同一独占设备上连续采集同一组的 10 个样本。

---

## 8. 结果记录与作业回答

CPU 正式实验已经完成，完整环境、命令、原始统计、warmup 对照、内存边界和可直接提交的 (b)/(c) 答案见 [01_02 CPU 端到端 Benchmark 实验报告](./01_02_cpu_benchmark_experiment_report.md)。

在固定 `batch_size=4`、`context_length=512`、`dtype=float32`、`warmup_steps=5`、`measurement_steps=10` 的条件下，`small`、`medium` 和 `large` 均完成 full mode；`xl` 因内存边界只完成 forward+backward，`10b` 只完成 forward。报告不会把这两组降级结果混入完整训练横向对比，也不会通过缩小 batch、sequence 或模型规模伪装为原实验。

CPU 数据用于验证端到端执行链路和回答 CPU 上的 warmup 现象，不能替代 GPU 数据。独占 GPU 上的 handout 正式实测仍待完成。

---

## 9. 实现检查清单

- [x] 使用给定超参数和内置 preset 初始化 Assignment 1 模型。
- [x] 生成固定随机输入与目标，数据生成不计时。
- [x] 支持 forward、forward+backward、完整训练三种模式。
- [x] 支持任意非负 warmup 次数与正数 measurement 次数。
- [x] CUDA 阶段结束和每个 step 结束后同步。
- [x] 报告 10 次样本的均值、总体标准差、最小值、最大值和原始值。
- [x] JSON 输出支持后续自动构造表格。
- [x] measurement loop 带 NVTX range，方便后续 `nsys` 复用。
- [x] CPU 冒烟测试、Ruff 与 ty 检查通过。
- [x] 在 CPU 上完成可行的正式实验、warmup 对照和实验报告。
- [ ] 在独占 GPU 上完成 handout (b)、(c) 的正式实测。
