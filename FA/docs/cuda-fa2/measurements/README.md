# CUDA FA2 测量索引

每个代码版本的测量单独归档到 `v0/`、`v1/`、`v2/` 等目录。目录名和文件名不使用日期；测量日期保留在文档与 `manifest.json` 中。不同版本各自记录源码、实际加载的二进制、运行环境、理论预期和原始结果。

## 1. 已归档测量

| 版本 | 代码版本与范围 | 阅读入口 | 状态 |
|---|---|---|---|
| V0 | custom 参考提交 `be1e825df74e09adb947fb90182805e2d846080f`；GTX 1060，FP32，`S=16384, d=64` | [版本清单](v0/manifest.json)、[性能分析](v0/perf-baseline.md)、[指标解释](v0/nvprof-readable.md)、[原始 benchmark 记录](v0/opt_routine1.md) | baseline 已归档；硬件计数器与时间线对比为 non-causal，原始 benchmark 文档另保留 causal 记录 |

V1 及后续版本在完成各自实验后添加到本表。通用工具说明见 [性能调优工具与学习路线](../profiling-tools.md)。

## 2. 每个测量目录的结构

```text
measurements/
├── README.md
└── v0/
    ├── README.md           阅读顺序和本次资料范围
    ├── manifest.json       代码、二进制、环境与数据哈希
    ├── opt_routine1.md     原始 benchmark 命令与输出
    ├── perf-baseline.md    性能分析与优化假设
    ├── nvprof-readable.md 指标定义和本次实测解读
    ├── data/              CSV、JSON、日志、资源与 SASS
    ├── assets/            SVG 图和原始截图
    ├── snapshot/          参考源码与实际测量二进制
    └── profiles/          原始 profiler 报告和已有分析导出
```

后续版本使用相同结构；相同版本有多轮独立采集时，在版本目录下用 `run1/`、`run2/` 区分，保留每轮条件与原始输出。

## 3. 优化前后比较

优化前先记录预期减少的工作量和成立条件；优化后分别核对机器指令或结构计数、硬件指标与无 profiler 延迟。完整方法见 [V0 的理论预期与实测对照](v0/nvprof-readable.md#theory-validation)。

结果归档时确认实际导入模块路径与哈希，不能只记录 Git HEAD。不同 shape、causal、精度、计时方案、工具版本与 GPU 条件分别保留各自口径；原始数据不因后续优化而覆盖。
