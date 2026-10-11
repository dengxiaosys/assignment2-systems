# FA2 CUDA 源码与构建

版本目录仅存放该版本的核心实现，`common/` 存放共用绑定；构建入口统一为 `build_extension.py`。

```text
fa2/
├── build_extension.py
├── common/
│   └── binding.cpp
├── v0/
│   └── fa2_forward.cu
└── v1/
    └── fa2_forward.cu
```

在 `FA` 目录中选择版本编译，默认版本为 V1：

```bash
TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=1 \
python cpp_cuda/fa2/build_extension.py --version v1 build_ext --inplace
```

`--version v0` 生成 `src.cuda_fa2_extension`，`--version v1` 生成 `src.cuda_fa2_v1_extension`。两版使用独立模块名，运行选择分别为 `--impl cuda_fa2` 与 `--impl cuda_fa2_v1`。

完整步骤见 [V1 编译执行文档](../../docs/cuda-fa2/measurements/v1/build-and-run.md)。性能资料见 [测量索引](../../docs/cuda-fa2/measurements/README.md)：每版 `snapshot/` 保留核心 CUDA 实现，公共绑定、脚本与测试直接引用工作区；实际被测扩展独立归档。常规优化复用 V0 已归档基线，仅测量新版本。
