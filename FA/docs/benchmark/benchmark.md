# Attention Forward Benchmark

## 1. 测量目标

对二维单头 self-attention 做 FP32 forward 对照：

| `--impl` | 实现 | 完整物化 `S/P` |
|---|---|---|
| `native` | `softmax(QKᵀ / sqrt(d))V` 的直接 PyTorch 实现 | 是 |
| `efficient` | 强制使用 PyTorch memory-efficient SDPA | 否 |

两条路径使用相同的 `(S, d)` 输入、causal 规则、精度、warmup 和计时方式。
PyTorch CUDA FlashAttention 后端不支持本实验固定的 FP32，因此不在这个对照中。

运行入口为
[scripts/benchmark.py](../../scripts/benchmark.py)，
通用计时逻辑位于
[src/measurement.py](../../src/measurement.py)。

## 2. 运行

以下命令均在 `FA` 目录执行：

```bash
cd /home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA

# 大规模原生基线；避免创建巨大的 CPU FP64 参考中间量
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark \
  --device cuda --impl native \
  --seq-len 16384 --head-dim 64 --no-verify

# 相同 FP32 输入上的 memory-efficient SDPA
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark \
  --device cuda --impl efficient \
  --seq-len 16384 --head-dim 64 --no-verify

# CPU 冒烟验证
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark \
  --device cpu --impl native \
  --seq-len 257 --head-dim 64 \
  --warmup 2 --iterations 5 --repeats 3
```

添加 `--causal` 可测 causal attention。使用
`--output-json ./benchmark_results/result.json` 可保存结果；目标目录会自动创建。
GPU 不可用或指定 fused 后端不受支持时直接报错，不回退到 CPU、math 或其他后端。

## 3. 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--device` | `cuda` | `cpu` 或 `cuda`；`efficient` 只允许 CUDA |
| `--impl` | `native` | `native` 或 `efficient` |
| `--seq-len` | `16384` | 序列长度 `S` |
| `--head-dim` | `64` | 特征维度 `d` |
| `--causal` | 关闭 | 使用包含对角线的下三角可见性 |
| `--warmup` | `5` | 正式计时前的 forward 次数 |
| `--iterations` | `20` | 每个 repeat 内连续执行的 forward 次数 |
| `--repeats` | `5` | 独立测量批次数 |
| `--threads` | `1` | PyTorch CPU 线程数 |
| `--seed` | `0` | 输入随机种子 |
| `--no-verify` | 关闭 | 跳过 CPU FP64 参考校验 |
| `--output-json` | 无 | 可选 JSON 输出路径 |

## 4. 计时口径

每个 sample 是一批 `iterations` 次调用的平均值：

```text
sample_ms = 一批调用的总耗时 / iterations
```

脚本执行 `repeats` 批，然后对 sample 列表计算 mean 和 median。因此总正式计时约为：

```text
单次 forward 耗时 × iterations × repeats
```

例如单次 forward 为 `1.5686 s`，默认 `iterations=20, repeats=5`：

```text
1.5686 × 20 × 5 = 156.86 s
```

这还不包括 warmup、输入生成、CUDA 初始化、可选正确性校验和额外一次峰值显存测量。
大规模运行若未加 `--no-verify`，还会在单线程 CPU 上计算完整 FP64 参考 attention，
可能显著增加总运行时间和内存占用。

计时前创建输入并关闭 TF32。后端能力预检、输入生成、CPU 参考计算和数据拷贝不进入
正式计时。完整 eager forward 的参数校验、张量分配、算子发射与计算都计入。

## 5. 输出字段

| 字段 | 含义 |
|---|---|
| `implementation` / `backend` | 选择的实现和被强制启用的后端 |
| `qkv_shape` / `dtype` | 输入 shape 和固定 FP32 精度 |
| `wall_ms_samples` | 每批的单次 forward 平均 wall time |
| `cuda_event_ms_samples` | 每批的单次 forward 平均 CUDA Event 时间 |
| `*_mean` / `*_median` | sample 列表的均值和中位数 |
| `verified` | 是否执行 CPU FP64 参考校验 |
| `max_abs_error_vs_cpu_fp64` | 相对参考结果的最大绝对误差 |
| `native_s_plus_p_bytes` | 同 shape 下 native 两张完整矩阵的理论总字节数 |
| `materializes_full_attention_matrices` | 当前实现是否完整保存 `S/P` |
| `peak_additional_cuda_allocated_bytes` | 相对已存在输入的单次 forward 额外峰值活跃分配 |

CUDA Event 测量的是当前 stream 上 start/end Event 之间的时间，包含期间的 GPU 空闲；
它不是某一个 kernel 的耗时，也不等于 kernel duration 的简单相加。

## 6. 显存与复杂度

- 两次矩阵乘的总算术量为 `Θ(S²d)`，约 `4S²d` FLOPs。
- native softmax 为 `Θ(S²)`，完整 `S/P` 存储为 `Θ(S²)`。
- FP32 native 的两张矩阵合计 `8S²` bytes；`S=16384` 时为 2 GiB。
- memory-efficient 路径仍有 `Θ(S²d)` 算术量，但通过分块矩阵乘和 online softmax
  避免写出完整 `S/P`；具体 workspace 由后端决定。
- 峰值分配统计包含输出、mask 和算子临时量，不等于理论 `S/P` 字节数，也不是
  `nvidia-smi` 展示的进程总显存。

## 7. 正确性

默认使用
[src/verification.py](../../src/verification.py)
生成 CPU FP64 math SDPA 参考，FP32 校验容差为 `rtol=atol=2e-5`。
融合实现的运算顺序不同，不要求与 native 逐位相同。
