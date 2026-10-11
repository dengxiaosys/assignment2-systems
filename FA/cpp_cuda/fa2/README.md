# FA2 CUDA 源码与构建

版本目录仅存放该版本的核心实现，`common/` 存放共用绑定；构建入口统一为 `build_extension.py`。

```text
fa2/
├── build_extension.py
├── common/
│   └── binding.cpp
├── v0/
│   └── fa2_forward.cu
├── v1/
│   └── fa2_forward.cu
├── v2/
│   └── fa2_forward.cu
├── v3/
│   └── fa2_forward.cu
└── v4/
    └── fa2_forward.cu
```

在 `FA` 目录中选择版本编译，默认版本为 V1：

```bash
TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=1 \
python cpp_cuda/fa2/build_extension.py --version v1 build_ext --inplace
```

`--version v0` 生成 `src.cuda_fa2_extension`，V1/V2/V3/V4分别生成对应的`src.cuda_fa2_v1_extension`至`src.cuda_fa2_v4_extension`。运行选择为`--impl cuda_fa2`、`--impl cuda_fa2_v1`、`--impl cuda_fa2_v2`、`--impl cuda_fa2_v3`、`--impl cuda_fa2_v4`。

V1 为四个独立 query warp / CTA，V2 为八个 query warp 共享 32-key tile。V3 使用 64 query / CTA、4×4 score/thread 的寄存器矩阵块、K/PV shared union，并控制 QK/PV 的展开范围。统一入口默认仍为 V1，构建新版本时显式传 `--version v2` 或 `--version v3`。

V4使用64×64 CTA、8×4 score/thread、stage8分阶段预取、QK/PV shared union和P chunk置换。head_dim及输入起始地址满足对齐条件时使用float4 global搬运，其余输入走scalar路径。GTX1060主形状S=16384,d=64在交替成对计时中快于efficient，d>64等性能边界见[V4报告](../../docs/cuda-fa2/measurements/v4/perf_v4.md)。

开发源码的全局常量和整数非类型模板参数使用 `UPPER_SNAKE_CASE`，例如 `WARP_SIZE/WARPS_PER_BLOCK/DIMS_PER_LANE`。局部位置索引以 `i` 结尾，例如 `qi/ki/warpi/lanei/ri/dimi`；运行时长度、步长和数值使用语义明确的小写名称，例如 `head_dim/stride/q_dim`，不因声明为 `const` 就改为大写。CUDA 内建变量、已有 V0 实现与历史测量快照保留原名称。

空行按逻辑阶段分组：host 入口依次为输入检查与 device guard、形状与输出分配、launch 参数与 stream、模板分派与 launch 检查；kernel 的索引与边界、状态初始化、QK、softmax、PV 等阶段也以空行隔开。组内保持连续，避免逐行插入空行。

完整步骤见 [V1 编译执行文档](../../docs/cuda-fa2/measurements/v1/build-and-run.md)、[V2 编译执行文档](../../docs/cuda-fa2/measurements/v2/build-and-run.md)、[V3 编译执行文档](../../docs/cuda-fa2/measurements/v3/build-and-run.md)、[V4编译执行文档](../../docs/cuda-fa2/measurements/v4/build-and-run.md)。性能资料见 [测量索引](../../docs/cuda-fa2/measurements/README.md)：每版 `snapshot/` 保留核心 CUDA 实现，公共绑定、脚本与测试直接引用工作区；实际被测扩展独立归档。常规优化复用已归档基线，仅测量新版本。
