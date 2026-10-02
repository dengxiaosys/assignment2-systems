# CS336 Assignment 2 笔记阅读顺序

## 编号规则

文件名采用 `章_节_主题.md`：

| 章号 | 主题 |
|---|---|
| `00` | 作业 handout 与环境 |
| `01` | 单卡性能测量与 profiling |
| `02` | 混合精度与编译 |
| `03` | Triton 与 FlashAttention |
| `04` | 分布式通信与训练 |
| `05` | FSDP 与并行策略 |
| `90` | 独立运维附录 |

后续笔记按 handout 的依赖顺序继续编号。

## 00 准备与参考

1. [Assignment 2 handout 提取稿](./cs336_assignment2_systems_extracted.md)：题目原文与接口要求，作为索引和参考。
2. [00_01 NVIDIA GeForce GTX 1060 6GB 硬件性能参考](./00_01_gtx_1060_6gb_hardware_reference.md)：仅介绍标准 6GB GDDR5 版本的架构、算力、显存、精度支持、功耗和硬件瓶颈。

## 01 单卡性能测量

1. [01_01 端到端 Benchmark：同步计时、预热与统计](./01_01_end_to_end_benchmarking_notes.md)：`benchmarking_script` 的实现、背景知识、实验命令、复杂度和结果记录表。
2. [01_02 CPU 端到端 Benchmark 实验报告](./01_02_cpu_benchmark_experiment_report.md)：完整训练与 warmup 对照数据、资源边界、结果解释及 handout (b)/(c) 答案。
3. [01_03 Nsight Systems 单 Profile 实验报告](./01_03_nsys_profile_analysis.md)：基于一个 GTX 1060 profile 回答 `nsys_profile` (a)-(e)，并记录阶段边界与 kernel 统计。
4. [01_04 NVTX Range 与 Nsight Systems 入门](./01_04_nvtx_range_and_nsys_guide.md)：解释 `benchmark_measurement` range、CUDA 异步执行、profile 采集和分析流程。
5. [01_05 Large 模型内存 Profiling 实验报告](./01_05_large_memory_profiling_report.md)：用 CPU RSS timeline 和 saved-tensor hooks 分析 large 模型在 `S=128/2048` 下的 forward、backward、optimizer、mixed precision、单 block residual 与 gradient 内存。

## 02 混合精度与编译

1. [02_01 PyTorch autocast 与混合精度训练详解](./02_01_pytorch_autocast_mixed_precision_guide.md)：解释逐算子 dtype 决策、转换时机、weight cache、autograd 保存张量、GradScaler、手工精度控制和本项目实际 dtype 路径。
2. [02_02 Mixed-Precision Accumulation 实验报告](./02_02_mixed_precision_accumulation_report.md)：复现四种 FP16/FP32 累加组合，分析输入量化、累加舍入、ULP、误差曲线与高精度 accumulator 的作用。
3. [02_03 Benchmarking Mixed Precision 实验报告](./02_03_benchmarking_mixed_precision_report.md)：在本机 CPU 上用 small 模型比较 FP32 与 BF16 autocast，覆盖 ToyModel dtype、LayerNorm、完整训练 step、梯度和 AdamW 状态。
4. [02_04 PyTorch 梯度累积详解](./02_04_gradient_accumulation_guide.md)：从 batch/sequence 维、上游梯度和 VJP 推导共享参数梯度，再解释 `.grad` 累积、大 batch 等价、AMP、DDP/FSDP 与常见错误。
5. [02_05 RMSNorm Autograd Saved Tensors 与 Operator Fusion 实验报告](./02_05_rmsnorm_autograd_saved_tensors_report.md)：使用 saved-tensor hooks 比较 eager 与 `torch.compile` RMSNorm 的保存/取回事件、logical bytes、唯一 storage、整体 VJP 和数值一致性。
6. [02_06 PyTorch saved_tensors_hooks 机制详解](./02_06_pytorch_saved_tensors_hooks_guide.md)：解释 pack/unpack 与 `SavedVariable` 生命周期、引用和 storage 统计、RSS 边界、loss 统计范围、引用环风险以及 offload/checkpoint/compile 的关系。
7. [02_07 Memory-Optimal Gradient Checkpointing 实验报告](./02_07_gradient_checkpointing_report.md)：推导单层与递归 checkpoint 的 memory/compute 复杂度，并在 36 层 large block stack 上比较无 checkpoint 与 `k=1/2/3` 的 saved storage、RSS 峰值和运行时间。

## 03 Triton 与 FlashAttention

1. [03_01 PyTorch Attention CPU Benchmark 实验报告](./03_01_pytorch_attention_cpu_benchmark_report.md)：在 20 GiB 地址空间限制下完成 naive attention 的 20 组 CPU benchmark，分析 forward/backward 耗时、OOM 边界、Autograd saved storage 的 $S^2$ 增长及 FlashAttention 的消除方法。
2. [03_02 FlashAttention-2 Forward 初学者教材](./03_02_flash_attention_2_beginner_textbook.md)：从 weighted sum、Triton block pointer 和 online softmax 开始，系统推导 FA2 forward、causal mask、数值稳定性以及 FA1 到 FA2 的工作划分改进。
3. [03_03 FlashAttention-2 Backward 与实现验证](./03_03_flash_attention_2_backward.md)：从普通 attention backward 和 Softmax VJP 开始，推导 $L$ 重建、$D$ 行归约、tile 梯度与两遍调度，并继续说明 Triton 映射、测试、benchmark 和练习。
4. [03_04 从 FlashAttention-1 到 FlashAttention-2](./03_04_flash_attention_1_to_2_evolution.md)：从 FA1 的 I/O-aware 基线出发，详解 FA2 的 non-matmul 优化、序列维并行、sliced-Q、前后向调度差异与性能边界。
5. [03_05 从 FlashAttention-2 到 FlashAttention-3](./03_05_flash_attention_2_to_3_evolution.md)：详解 Hopper 的 TMA/WGMMA、producer-consumer 与两级 GEMM-softmax overlap、FP8 布局和数值误差控制。

## 04 分布式通信与训练

1. [04_01 PyTorch All-Reduce 本地实验报告](./04_01_pytorch_all_reduce_demo_report.md)：用 4 个本地 CPU worker 和 Gloo 验证 `SUM all_reduce` 的原地更新与全 rank 一致性，并解释进程组、rank、collective、Gloo/NCCL、同步语义以及与 DDP 梯度同步的关系。
2. [04_02 All-Reduce 底层通信背景](./04_02_all_reduce_communication_background.md)：从 collective 语义、$\alpha$-$\beta$ 模型和 ring/tree 算法继续深入到 Gloo/NCCL 源码边界、PCIe/NVLink/RDMA 数据路径、DDP/FSDP 调度与 benchmark 方法。
3. [04_03 单机 All-Reduce Benchmark 实验报告](./04_03_single_node_all_reduce_benchmark_report.md)：完成 CPU/Gloo 下 1 MB–1 GB、2/4/6 进程的三轮实验，报告逐次最慢 rank 延迟、波动、带宽与资源占用，并提供 GPU/NCCL 复跑入口。
4. [04_04 Naive DDP 正确性实验报告](./04_04_naive_ddp_experiment_report.md)：实现初始化状态广播与 backward 后逐参数梯度平均，在 2-rank CPU/Gloo 环境连续五轮对比全局 batch 基线，并推导等价条件、通信复杂度和适用边界。
5. [04_05 Naive DDP Benchmark 实验报告](./04_05_naive_ddp_benchmark_report.md)：实现统一三策略 benchmark，记录逐参数同步的计时语义、CPU/Gloo smoke、`xl` 静态规模和双 GPU/NCCL 复跑边界。
6. [04_06 Flat Gradient Minimal DDP Benchmark 实验报告](./04_06_flat_gradient_ddp_benchmark_report.md)：实现单 flat buffer all-reduce 与 copy-back，对比 collective 固定成本、额外内存流量、正确性和 CPU/Gloo 实测。
7. [04_07 逐参数通信与反向计算重叠的 DDP 实验报告](./04_07_overlapped_individual_parameter_ddp_report.md)：实现 post-accumulate hook、异步 Work 与 finish 协议，推导正确性、collective 顺序和支持边界。
8. [04_08 Overlapped DDP Benchmark 实验报告](./04_08_overlapped_ddp_benchmark_report.md)：比较 naive、flat、overlap 三种策略，说明 exposed tail、CUDA/NCCL stream 证据标准、Nsight 复跑命令和硬件阻塞。
