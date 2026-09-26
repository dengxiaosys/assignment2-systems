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

## 02 混合精度与编译

1. [02_01 PyTorch autocast 与混合精度训练详解](./02_01_pytorch_autocast_mixed_precision_guide.md)：解释逐算子 dtype 决策、转换时机、weight cache、autograd 保存张量、GradScaler、手工精度控制和本项目实际 dtype 路径。
2. [02_02 Mixed-Precision Accumulation 实验报告](./02_02_mixed_precision_accumulation_report.md)：复现四种 FP16/FP32 累加组合，分析输入量化、累加舍入、ULP、误差曲线与高精度 accumulator 的作用。
3. [02_03 Benchmarking Mixed Precision 实验报告](./02_03_benchmarking_mixed_precision_report.md)：在本机 CPU 上用 small 模型比较 FP32 与 BF16 autocast，覆盖 ToyModel dtype、LayerNorm、完整训练 step、梯度和 AdamW 状态。
4. [02_04 PyTorch 梯度累积详解](./02_04_gradient_accumulation_guide.md)：从 batch/sequence 维、上游梯度和 VJP 推导共享参数梯度，再解释 `.grad` 累积、大 batch 等价、AMP、DDP/FSDP 与常见错误。
