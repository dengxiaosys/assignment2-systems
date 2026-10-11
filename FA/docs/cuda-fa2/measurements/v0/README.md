# V0：FP32 custom attention baseline

本目录集中保存这批 baseline 测量的文档、原始结果、图表和代码证据。测量与分析日期为 2026-10-11；custom 参考提交为 `be1e825df74e09adb947fb90182805e2d846080f`。实际运行扩展的 SHA-256、环境和计时设置见 [manifest.json](manifest.json)。

## 1. 阅读顺序

1. [性能分析](perf-baseline.md)：结论、计时、硬件证据、同步与数据复用瓶颈。
2. [nvprof 指标解释](nvprof-readable.md)：GPU 基础、23 项指标、按 SM 观察和理论预期。
3. [原始 benchmark 记录](opt_routine1.md)：native、efficient、custom 的命令与输出。
4. [版本清单](manifest.json)：参考源码、实际二进制、环境、数据位置与哈希。

## 2. 本次资料范围

| 位置 | 内容 |
|---|---|
| [data/counters.json](data/counters.json) | `S=16384, d=64`、FP32、non-causal 的聚合计数器 |
| [data/evidence.json](data/evidence.json) | 原始报告哈希、时间线提取、资源与分析计数 |
| `data/` | nvprof CSV、采集日志、无 profiler 短复测、数值测试、SASS 与资源统计 |
| `assets/` | 三张 SVG 机制图和原始 benchmark 截图 |
| [snapshot/fa2_forward.cu](snapshot/fa2_forward.cu#L15-L82) | 从参考提交原样提取的源码；命名与当前开发文件可能不同 |
| [snapshot/cuda_fa2_extension.cpython-311-x86_64-linux-gnu.so](snapshot/cuda_fa2_extension.cpython-311-x86_64-linux-gnu.so) | 实际被测的预编译扩展 |
| `profiles/` | 本次分析引用的三份 `S=16384` Nsight 报告，以及已有 native SQLite 导出 |

profile 与分析主表采用 non-causal 条件；原始 benchmark 文档同时保留已有 causal 结果，不能与 non-causal 指标混作同一次采集。`data/nvprof-instance-capabilities.json` 记录的是后续能力查询，不是逐 SM 性能测量。

原始命令中的历史终端路径和工具日志按原样保留。文档链接与 `manifest.json` 的归档路径按本目录组织；`data/evidence.json` 的 Nsight 报告路径相对于该 JSON 所在目录。

后续优化结果放到同级 `v1/`、`v2/` 目录，参见 [测量索引与组织规范](../README.md)。
