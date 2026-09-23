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

## 01 单卡性能测量

1. [01_01 端到端 Benchmark：同步计时、预热与统计](./01_01_end_to_end_benchmarking_notes.md)：`benchmarking_script` 的实现、背景知识、实验命令、复杂度和结果记录表。
2. [01_02 CPU 端到端 Benchmark 实验报告](./01_02_cpu_benchmark_experiment_report.md)：完整训练与 warmup 对照数据、资源边界、结果解释及 handout (b)/(c) 答案。
