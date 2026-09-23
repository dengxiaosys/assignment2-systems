# CPU 端到端 Benchmark 实验报告

## 0. 结论摘要

本报告对应 handout 的 `benchmarking_script` (a)-(c)。在 CPU 上，`small`、`medium`、`large` 三个配置均完成了 5 次预热和 10 次完整训练测量，也完成了 `w=0,1,2,5` 的 warmup 对照实验。`xl` 完成 forward+backward，`10b` 完成 forward-only；受内存容量限制，后二者没有执行会创建完整 AdamW 状态的训练步骤。

CPU 实验验证了脚本、模型、loss、反向传播、优化器和统计链路，但不能替代 handout 要求的 GPU 性能结论。GPU 的异步执行、kernel 启动、显存分配和频率状态均与 CPU 不同，后续仍需在独占 GPU 上复跑。

---

## 1. 实验环境

### 1.1 硬件与软件

| 项目 | 配置 |
|---|---|
| CPU | 2 x Intel Xeon Platinum 8336C @ 2.30 GHz |
| 核心 | 56 个物理核，2 个 NUMA node，每核 1 线程 |
| 内存 | 109 GiB，总可用内存约 65 GiB（实验启动前） |
| 操作系统 | Linux 5.4.143.bsk.8-amd64 x86_64 |
| PyTorch | 2.11.0+cu130 |
| dtype | FP32 |
| 实验日期 | 2026-09-23 |

### 1.2 固定条件

所有正式测量固定使用：

- `batch_size=4`
- `context_length=512`
- `vocab_size=10000`
- `measurement_steps=10`
- `seed=0`
- `OMP_NUM_THREADS=50`
- `MKL_NUM_THREADS=50`
- `taskset -c 0-49`

这样将计算限制在 50 个 CPU 核上，给系统保留 6 个核。两个 CPU socket 都参与计算，因此结果包含跨 NUMA node 调度和共享机器负载带来的波动，不应视为严格隔离环境下的 CPU 峰值性能。

### 1.3 复现命令

完整训练实验采用以下命令，其中 `size` 取 `small`、`medium` 或 `large`：

```bash
OMP_NUM_THREADS=50 MKL_NUM_THREADS=50 taskset -c 0-49 /usr/bin/time -v uv run python benchmark.py --model-size "$size" --mode full --device cpu --dtype float32 --warmup-steps 5 --measurement-steps 10 --output-json "benchmark_results/cpu/${size}_w5.json"
```

warmup 对照实验只将 `--warmup-steps` 改为 `0`、`1`、`2` 和 `5`，并且每组都启动独立进程，避免后运行的组继承前一组的内存池和运行时状态。`xl` 使用 `--mode forward-backward`，`10b` 使用 `--mode forward`。

原始 JSON 和 `/usr/bin/time -v` 输出保存在本地 `benchmark_results/cpu/`。该目录已加入 `.gitignore`，避免把机器相关的大量实验产物纳入版本控制。

---

## 2. 五次预热后的正式结果

### 2.1 完整训练结果

表中数值均为 10 次 measurement 的“均值 ± 总体标准差”，单位为毫秒。`total` 还包含单独计时的 loss。

| 配置 | 参数量 | forward (ms) | loss (ms) | backward (ms) | optimizer (ms) | total (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `small` | 128.63M | 457.588 ± 121.259 | 11.567 ± 3.223 | 1853.362 ± 705.051 | 105.435 ± 111.282 | 2428.003 ± 805.802 |
| `medium` | 423.18M | 2501.541 ± 965.578 | 15.762 ± 9.788 | 7805.593 ± 1930.809 | 708.489 ± 997.638 | 11031.438 ± 2414.980 |
| `large` | 969.41M | 6271.720 ± 2660.195 | 31.259 ± 16.997 | 14799.688 ± 1998.194 | 1371.282 ± 1271.659 | 22474.004 ± 3991.225 |

三个配置中 backward 都是主要耗时，约为 forward 的 2.36-4.05 倍。原因是 backward 不仅要沿计算图传播激活梯度，还要为各层计算参数梯度；实际比例还受到矩阵形状、CPU kernel、内存带宽和线程调度影响。

结果的波动不能视为“小”。`total` 的变异系数分别为 33.19%、21.89% 和 17.76%；部分 optimizer 样本还受到线程调度和内存访问抖动影响，标准差接近或超过均值。该现象说明共享双路 CPU 不是稳定的性能基准环境，但不影响验证执行链路和比较数量级。

### 2.2 因内存约束而降级的结果

| 配置 | 实际 mode | 参数量 | forward (ms) | backward (ms) | total (ms) | 峰值 RSS |
|---|---|---:|---:|---:|---:|---:|
| `xl` | `forward-backward` | 3.41B | 12459.476 ± 1193.265 | 27415.451 ± 2661.358 | 39907.033 ± 2698.482 | 46.64 GiB |
| `10b` | `forward` | 12.83B | 44126.510 ± 1606.595 | 不适用 | 44126.607 ± 1606.600 | 49.23 GiB |

这些结果证明大模型的相应执行路径可以在 CPU 上跑通，但 mode 不同，不能与上一表的完整训练总耗时直接比较。

`xl` 的 FP32 参数约占 12.69 GiB。若执行 AdamW，参数、梯度、一阶矩和二阶矩仅静态状态下界就约为 50.77 GiB，尚未包括激活、临时张量和框架开销；在 forward+backward 已达到 46.64 GiB RSS 的情况下继续创建 AdamW 状态，预计至少再增加 25.38 GiB，超过实验启动时约 65 GiB 的可用内存。

`10b` 的 FP32 参数本身约占 47.81 GiB，参数加梯度约需 95.62 GiB，完整训练静态状态下界约为 191.22 GiB。因此本机不能安全执行其 backward 或 full mode。这里保留原始 batch、sequence 和模型规模，不通过缩小实验参数伪装成 handout 的正式结果。

### 2.3 进程资源记录

| 配置 | mode | 墙钟时间 | 平均 CPU 利用率 | 峰值 RSS | major page faults |
|---|---|---:|---:|---:|---:|
| `small` | `full` | 1:01.02 | 4448% | 6.76 GiB | 0 |
| `medium` | `full` | 3:20.98 | 4483% | 17.16 GiB | 0 |
| `large` | `full` | 6:30.97 | 4323% | 33.78 GiB | 94 |
| `xl` | `forward-backward` | 11:50.94 | 4121% | 46.64 GiB | 14 |
| `10b` | `forward` | 13:51.86 | 3567% | 49.23 GiB | 2103 |

`/usr/bin/time` 的 CPU 利用率以单核 100% 为基准，因此 4448% 约等于平均占用 44.48 个核。模型增大后利用率下降且 major page faults 增多，表明内存压力和数据移动对扩展性产生了更明显的限制。

---

## 3. Warmup 对照实验

### 3.1 总耗时与变异系数

每个单元格为 `total mean ± std (CV)`，单位为毫秒。

| 配置 | `w=0` | `w=1` | `w=2` | `w=5` |
|---|---:|---:|---:|---:|
| `small` | 4519.810 ± 4256.830 (94.18%) | 3841.192 ± 2002.712 (52.14%) | 3309.257 ± 1137.121 (34.36%) | 2428.003 ± 805.802 (33.19%) |
| `medium` | 19591.005 ± 10576.774 (53.99%) | 10441.322 ± 2483.813 (23.79%) | 14400.593 ± 3663.424 (25.44%) | 11031.438 ± 2414.980 (21.89%) |
| `large` | 26481.673 ± 5770.681 (21.79%) | 29082.161 ± 5523.457 (18.99%) | 21662.672 ± 4372.051 (20.18%) | 22474.004 ± 3991.225 (17.76%) |

### 3.2 现象解释

`small` 的趋势最清晰：warmup 从 0 增加到 5 时，均值和 CV 都持续下降。首次迭代会包含线程池初始化、内存页首次触达、算子库内部初始化，以及 AdamW 一阶矩和二阶矩的首次分配；没有预热时，这些一次性成本被计入 measurement。

`medium` 和 `large` 的均值没有随 warmup 次数严格单调下降。例如 `medium w=1` 快于 `w=2`，`large w=1` 反而最慢。这并不否定 warmup 的作用，而是说明 CPU 结果还受到共享负载、NUMA 调度、频率变化、内存页状态和缓存状态影响。每组只有 10 个样本，在 18%-54% 的 CV 下，均值差异中混有较强噪声。

1-2 次 warmup 仍可能与 5 次不同，因为“初始化路径已经执行”不等于“系统达到稳态”。线程调度、缓存、页映射和内存分配器可能需要多轮才稳定；在共享 CPU 上，即使增加 warmup 也不能消除外部负载造成的波动。

---

## 4. Handout 回答

### 4.1 (b) 1-2 句答案

**原问题**

> Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?
>
> **Deliverable:** A 1-2 sentence response with your timings.

**回答**

在 2 x Xeon Platinum 8336C、FP32、5 次预热和 10 次测量下，`small/medium/large` 的 forward 分别为 457.588 ± 121.259、2501.541 ± 965.578、6271.720 ± 2660.195 ms，backward 分别为 1853.362 ± 705.051、7805.593 ± 1930.809、14799.688 ± 1998.194 ms，optimizer 分别为 105.435 ± 111.282、708.489 ± 997.638、1371.282 ± 1271.659 ms。测量波动较高，完整 step 的 CV 为 17.76%-33.19%；`xl` 和 `10b` 因本机内存不足未完成 full mode，CPU 数据只用于链路验证，GPU 正式结果仍待补充。

### 4.2 (c) 2-3 句答案

**原问题**

> One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?
>
> **Deliverable:** A 2-3 sentence response.

**回答**

不预热时，`small/medium/large` 的完整 step 分别为 4519.810 ± 4256.830、19591.005 ± 10576.774、26481.673 ± 5770.681 ms，其中前两者的 CV 达 94.18% 和 53.99%，因为首次测量吸收了线程池、内存页、算子库和 AdamW 状态的初始化成本。1-2 次预热仍可能与 5 次不同，因为初始化完成后缓存、页映射、线程调度和频率状态未必已稳定；本次共享双路 CPU 上还观察到非单调变化，说明系统噪声不能通过少量 warmup 完全消除。

---

## 5. 有效性边界与后续实验

1. 本报告是 CPU 正确性与资源边界报告，不是 GPU 性能报告；不能据此推断 CUDA kernel 时间。
2. `xl` 和 `10b` 的降级 mode 只验证可执行路径，不回答完整训练耗时。
3. GPU 复跑必须保持 `B=4`、`S=512`、`V=10000`、FP32、`w=5`、`n=10`，OOM 时应原样记录，不能缩小参数后混入同一结果表。
4. GPU warmup 对照应在同一型号的独占 GPU 上逐组启动独立进程，并使用 `torch.cuda.synchronize()` 保证计时边界正确。
5. 若要得到更稳定的 CPU 基线，应独占机器、固定 NUMA 内存策略与 CPU 频率，并提高 measurement 次数后报告置信区间。
