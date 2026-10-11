# 1. V4 编译、验证与测量

## 1.1 环境与实测版本

本版在GTX1060、sm61、CUDA12.8、PyTorch2.6.0+cu124上测量，使用`nanovllm`环境中的Python3.11。项目默认Python约束与该历史测量环境不同；复现本次扩展应使用下面的明确解释器路径。编译与运行均限制CPU并行度为1，GPU实验顺序执行。

源码为[核心CUDA实现](file:///home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA/cpp_cuda/fa2/v4/fa2_forward.cu)，最终实测版本为run20；[manifest](manifest.json)记录实际二进制哈希、源码哈希、资源、计数器与原始数据位置。工作区开发扩展`src/*.so`不追踪，实测二进制保存在`binaries/`。

以下命令在`stanford_cs336/assignments/assignment2-systems/FA`执行。本会话的CUDA初始化需要已授权的sudo，故命令保留该运行身份。普通可直接访问GPU的环境可以去掉sudo，其他参数不变。

## 1.2 编译

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin TORCH_CUDA_ARCH_LIST=6.1 \
  MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  cpp_cuda/fa2/build_extension.py --version v4 build_ext --inplace \
  --build-temp /tmp/fa2-v4-build/temp --build-lib /tmp/fa2-v4-build/lib
```

编译选项为`-O2 -lineinfo --ptxas-options=-v`，没有`--use_fast_math`。只有softmax显式采用PTX `ex2.approx.f32`，未设置`.ftz`。主aligned模板157 registers/thread、21760 B shared；全部模板无stack/local/spill。资源必须由实际生成的二进制核对，不能只依据源码估算。

## 1.3 数值与同步验证

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m tests.test_cuda_fa2_forward --impl cuda_fa2_v4 --device cuda \
  --long-sequence --output-json docs/cuda-fa2/measurements/v4/data/row-validation.json
```

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m tests.test_cuda_fa2_forward --impl cuda_fa2_v4 --device cuda
```

分别将`memcheck`替换为`synccheck`和`racecheck`。sanitizer执行14项常规测试，不执行S=16384长序列测试。最终独立数值执行15项测试，包括长序列两种模式各10行抽样；报告明确没有验证完整S×S参考。

测试覆盖d=1..128、奇数维度、key/query尾部、因果对角线、最大值跨tile改变、正负抵消、次正规概率乘大V，以及连续但起始地址不对齐的输入。rtol=atol=2e-5。

## 1.4 无 profiler 计时

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.benchmark_attention --impl cuda_fa2_v4 --device cuda \
  --seq-len 16384 --head-dim 64 --no-causal --no-verify \
  --output-json docs/cuda-fa2/measurements/v4/data/benchmark-custom.json
```

该命令为5次warmup、每批20次、5批；efficient对照用`--impl efficient`。`--no-verify`仅跳过本次计时中的完整FP64参考，独立数值验证另行执行。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.benchmark_attention_comparison --suite paired \
  --output-json docs/cuda-fa2/measurements/v4/data/benchmark-paired.json
```

`paired`使用相同输入、seed0/1/7、两种模式、各5组交替执行顺序；每批5次warmup和20次计时。`--suite matrix`覆盖24个形状×两种模式，每组3轮、10次计时。两套比较均先对两种实现做CPU FP64抽样行校验；所有原始样本保存于JSON。动态时钟没有锁频，跨轮温度变化可能改变绝对时间，比较应优先使用同轮比值。

## 1.5 计数器、时间线与SASS

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/nvprof --profile-from-start off --csv \
  --metrics achieved_occupancy,eligible_warps_per_cycle,issue_slot_utilization,stall_memory_dependency,stall_inst_fetch \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.profile_attention --impl cuda_fa2_v4 --no-causal
```

完整29项名称与结果见manifest的`profile.metrics`，原始CSV保存在[data](data/nvprof-custom-s16384-d64.csv)。采集一个forward调用，counter replay不用于延迟比较。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/nvprof --profile-from-start off --csv --print-gpu-trace \
  --export-profile docs/cuda-fa2/measurements/v4/profiles/forward.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.profile_attention --impl cuda_fa2_v4 --no-causal
```

```bash
cuobjdump --dump-resource-usage docs/cuda-fa2/measurements/v4/binaries/cuda_fa2_v4_extension.cpython-311-x86_64-linux-gnu.so
cuobjdump --dump-sass docs/cuda-fa2/measurements/v4/binaries/cuda_fa2_v4_extension.cpython-311-x86_64-linux-gnu.so
```

资源与静态指令统计集中存于manifest，不重复追踪完整SASS和资源文本。历史基线V0–V3与efficient库身份直接引用既有归档，不复制或重跑旧版本。消融run21只强制走scalar路径，结束后恢复run20源码与二进制；最终工作区哈希应与manifest一致。
