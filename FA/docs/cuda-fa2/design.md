# FP32 CUDA FlashAttention 基线实现

## 1. 实现目标

本目录提供一个正确性优先的 CUDA Attention 基线，后续将通过可独立测量的步骤逐渐优化至 FlashAttention-2 风格实现。

当前版本已经具备 FlashAttention 的核心显存算法：

- 精确计算 Scaled Dot-Product Attention，不使用近似算法。
- 使用 Online Softmax。
- 不物化 `(S, S)` 分数矩阵和概率矩阵。
- 支持 Causal 与 Non-Causal Forward。

当前版本有意不直接加入生产级 FA2 的性能优化。这样可以分别对后续每一项改动进行 Benchmark 和 Profiling，判断哪些优化真正有效。

## 2. 支持范围

| 项目 | 当前支持 |
|---|---|
| Q/K/V/O Shape | `(S, d)` |
| 数据类型 | FP32 |
| Attention 类型 | 单头 Self-Attention |
| Tensor 要求 | 连续 CUDA Tensor |
| Head Dimension | `1 <= d <= 128` |
| 计算方向 | Forward Only |
| Causal | 支持 |

当前不支持 Backward、Dropout、任意 Attention Mask、GQA、KV Cache、FP16/BF16 以及 Tensor Core 路径。

相关文件：

- [binding.cpp](../../cpp_cuda/fa2/binding.cpp)：暴露 C++ 接口。
- [fa2_forward.cu](../../cpp_cuda/fa2/fa2_forward.cu)：实现基线 CUDA Kernel。
- [build_extension.py](../../cpp_cuda/fa2/build_extension.py)：使用不依赖 Ninja 的方式构建扩展，详细说明见 [extension-build.md](./extension-build.md)。
- [cuda_fa2_forward.py](../../src/cuda_fa2_forward.py)：提供 Python Wrapper。

## 3. 基线并行划分

Kernel 为每个 Query 行分配一个 128-Thread CTA：

```text
grid.x  = S
block.x = 128
```

处理每个可见 Key 时：

1. 各线程计算 `Q_i * K_j` 的一个元素。
2. 使用 Shared-Memory Tree Reduction 得到完整点积。
3. 由 Thread 0 更新 Online-Softmax 状态。
4. 各线程使用 `V_j` 更新一个输出维度。

每个 Query 行都会从 Global Memory 重新读取 K/V。当前还没有跨 Query 行复用、向量化加载或 Shared-Memory K/V Tiling。

## 4. Online Softmax

每个 Query 行维护以下状态：

| 符号 | 含义 |
|---|---|
| `m` | 已处理 Attention Score 的最大值 |
| `l` | Softmax 的累计归一化系数 |
| `o` | 尚未归一化的输出累加器 |

处理新的 Score `x` 时：

```text
m_new = max(m, x)
alpha = exp(m - m_new)
beta  = exp(x - m_new)
l     = alpha * l + beta
o     = alpha * o + beta * V_j
m     = m_new
```

处理完所有可见 Key 后写出 `o / l`。

复杂度：

- 算术复杂度仍为 `Θ(S²d)`。
- 全局辅助存储由 `Θ(S²)` 降为 `Θ(Sd)`。
- 不再向 Global Memory 写入完整 Score 和 Probability 矩阵。

## 5. 后续优化路线

基线有意保留下列可独立测量的优化点：

| 阶段 | 优化项 | 主要观察指标 |
|---:|---|---|
| 1 | 使用 Warp Shuffle Reduction 替代 Shared-Memory Tree Reduction | Barrier 数量、Kernel Duration |
| 2 | 每个 CTA 并行处理多个 Query 行 | CTA 数量、Occupancy、K/V 复用率 |
| 3 | 将 K/V Tile 加载到 Shared Memory | Global-Memory Traffic、带宽利用率 |
| 4 | 使用向量化 Q/K/V Global Load | Load 指令数、内存事务 |
| 5 | 使用 Double Buffering 重叠加载与计算 | Stall、指令流水 |
| 6 | 调节 Q/K Tile 大小 | Registers、Shared Memory、Occupancy |
| 7 | 为常见 Head Dimension 提供模板特化 Kernel | 分支与循环开销 |
| 8 | 增加低精度 Tensor Core 路径 | Tensor Core 利用率、吞吐量 |

完成测量后再决定是否保留该优化，避免只凭经验判断。

## 6. 测量口径

每一步优化都应记录：

- 相对 CPU FP64 参考结果的数值误差。
- CUDA Event Forward 延迟。
- 单个 Kernel Duration。
- Kernel Launch 间隙。
- Global-Memory Traffic。
- Occupancy、Register 和 Shared-Memory 使用量。
- 额外峰值显存。

Benchmark 与 Profiling 必须保持相同的 `S`、`d`、FP32、Causal 设置和 Warmup 次数。

## 7. 构建

当前构建不依赖 Ninja。在 `FA` 目录执行：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=6.1 \
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py build_ext --inplace
```

构建产物 `.so` 写入 `src/` Python Package，编译中间文件写入 Setuptools 的普通 `build/` 目录；两者均被 Git 忽略。

运行测试、Benchmark 或 Profiling 前需要先完成构建。CUDA 源码使用 `-lineinfo` 编译，因此可以在 Nsight Systems 中识别 `fa2_forward_fp32_kernel`。
