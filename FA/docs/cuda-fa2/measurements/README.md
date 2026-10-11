# CUDA FA2 测量索引

每个代码版本的测量单独归档到 `v0/`、`v1/`、`v2/` 等目录。目录名和文件名不使用日期；测量日期保留在文档与 `manifest.json` 中。不同版本各自记录源码、实际加载的二进制、运行环境、理论预期和原始结果。

## 1. 版本与测量状态

| 版本 | 代码版本与范围 | 阅读入口 | 状态 |
|---|---|---|---|
| V0 | custom 参考提交 `be1e825df74e09adb947fb90182805e2d846080f`；GTX 1060，FP32，`S=16384, d=64` | [版本清单](v0/manifest.json)、[性能分析](v0/perf_v0.md)、[指标解释](v0/nvprof-readable.md)、[原始 benchmark 记录](v0/opt_routine1.md) | baseline 已归档；硬件计数器与时间线对比为 non-causal，原始 benchmark 文档另保留 causal 记录 |
| V1 | 以 V0 为父版本；warp 内归约、一个 warp 一个 query、四个 query / CTA | [初学者 FAQ](v1/faq.md)、[编译执行](v1/build-and-run.md)、[实测分析](v1/perf_v1.md)、[版本清单](v1/manifest.json)、[优化前设计](v1/optimization-design.md) | 已编译、验证与采集；主形状 non-causal 348.178 ms，相比 V0 原基线快 8.393 倍 |

后续版本先添加优化前设计，再补充各自实验与证据。通用工具说明见 [性能调优工具与学习路线](../profiling-tools.md)。

性能报告统一命名为 `perf_v0.md`、`perf_v1.md`、`perf_v2.md` 等，与目录版本对应；V0 的基线角色在正文中说明。

## 2. 每个测量目录的结构

```text
measurements/
├── README.md
├── v0/
│   ├── manifest.json       代码、二进制、环境与数据哈希
│   ├── opt_routine1.md     原始 benchmark 命令与输出
│   ├── perf_v0.md          V0 性能分析与优化假设
│   ├── nvprof-readable.md  指标定义和本次实测解读
│   ├── data/              CSV、JSON、日志、资源与 SASS
│   ├── assets/            SVG 图和原始截图
│   ├── snapshot/          参考源码与实际测量二进制
│   └── profiles/          原始 profiler 报告；SQLite 导出不追踪
└── v1/
    ├── faq.md                  设备、kernel 发射、shared 与同步背景
    ├── optimization-design.md  优化前数学模型、实现规格与验证协议
    ├── build-and-run.md        构建、测试、benchmark 与 profile 复现
    ├── perf_v1.md              V1 性能与计数器分析
    ├── manifest.json          实际源码、扩展与测量身份
    ├── data/                  编译、测试、计时、CSV、资源与 SASS
    ├── snapshot/              仅核心 fa2_forward.cu
    ├── binaries/              实际被测扩展
    ├── profiles/              nvprof 原始报告
    └── assets/                优化设计的 SVG 示意图
```

FA2 的 [源码目录](../../../cpp_cuda/fa2/README.md) 使用 `common/`、`v0/`、`v1/` 与单一 `build_extension.py --version` 入口。每版 `snapshot/` 仅保留核心 CUDA 实现，公共绑定、脚本和测试直接引用工作区；实际扩展单独归档，V0 历史目录保持原有二进制位置。多轮独立采集用 `run1/`、`run2/` 区分，保留各轮条件与原始输出。

## 3. 优化前后比较

优化前先记录预期减少的工作量和成立条件；优化后分别核对机器指令或结构计数、硬件指标与无 profiler 延迟。完整方法见 [V0 的理论预期与实测对照](v0/nvprof-readable.md#theory-validation)。

结果归档时确认实际导入模块路径与哈希，不能只记录 Git HEAD。不同 shape、causal、精度、计时方案、工具版本与 GPU 条件分别保留各自口径；原始数据不因后续优化而覆盖。

常规优化只执行新版本，V0 的正式 benchmark、原始样本与 counters 直接引用既有归档，不复制到每个新版本，也不每次重跑。主形状 V0 的正式均值为 2922.2971875 ms、中位数为 2916.776953125 ms，见 [V0 manifest](v0/manifest.json)；均值比均值、中位数比中位数。只有确需环境校准时才另开独立 V0 实验，已经产生的复跑记录可保留为补充证据。

## 4. Git 跟踪范围

Git 保留源码、文档与图示，以及已归档的实际测量证据：CSV/JSON、编译与验证日志、SASS/资源统计、原始 `.nsys-rep/.nvvp` 报告和实测扩展二进制。这些文件固定某次实验的结果或执行身份；重新运行可能产生不同结果。

`FA/build/`、`FA/profiles/` 临时采集目录、`src/*.so` 开发构建产物、Python 缓存、散落的 object/static library 和 Nsight SQLite 导出均不追踪。SQLite 可从原始 `.nsys-rep` 重新生成，方法见 [V0 资料说明](v0/README.md)。测量目录对日志和归档 `.so` 有定向例外，避免将实验依据一起忽略。
