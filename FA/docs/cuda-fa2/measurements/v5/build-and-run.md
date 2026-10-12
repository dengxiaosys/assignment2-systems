# 1. V5 编译、验证与测量

## 1.1 环境与版本身份

本次测量使用GTX1060（sm61）、CUDA Toolkit12.8、PyTorch2.6.0+cu124和CPython3.11。最终源码位于[版本目录](../../../../cpp_cuda/fa2/v5/fa2_forward.cu)，实际扩展、源码快照、环境和全部证据哈希见[manifest](manifest.json)。编译和CPU FP64参考均限制单线程，所有GPU任务顺序执行，未调整频率或功耗。

以下命令在`stanford_cs336/assignments/assignment2-systems/FA`执行。该会话CUDA设备初始化使用已授权的sudo；具备普通GPU访问权限的环境可使用相同命令参数直接运行。开发扩展`src/*.so`不追踪，实际被测扩展保存在`binaries/`，构建中间产物保存在`/tmp/fa2-v5-build/`。

## 1.2 编译与头文件依赖

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin TORCH_CUDA_ARCH_LIST=6.1 \
  MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  cpp_cuda/fa2/build_extension.py --version v5 build_ext --inplace --force \
  --build-temp /tmp/fa2-v5-build/temp --build-lib /tmp/fa2-v5-build/lib
```

默认编译参数为`-O2 -lineinfo --ptxas-options=-v`，不使用`--use_fast_math`、寄存器强制上限或SASS二进制patch。`build_extension.py`将全部版本`.cuh`列入`depends`；实验仍使用`--force`确保头文件改动实际进入二进制。run12曾因头文件未触发重编译而无效，不能将其旧二进制时长归因于新方案。

主路径由`fixed_prefetch.cuh`、`pack_qk.cuh`、`score_bounds.cuh`、`register_mma.cuh`和`combine.cuh`组成；`maxwell_d64.cuh`负责规整d64的范围外在线softmax。`fa2_forward.cu`保留输入校验、host分派与其他d/尾部/不对齐通用路径。主路径每CTA48 KiB shared、128threads，实际寄存器与spill由manifest的`resources`记录。

## 1.3 数值与sanitizer

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m tests.test_cuda_fa2_forward --impl cuda_fa2_v5 --device cuda \
  --long-sequence --output-json docs/cuda-fa2/measurements/v5/data/row-validation.json
```

最终独立数值测试为17项，包括d=1..128、query/key尾部、因果对角、不对齐storage offset、正负抵消、跨tile最大值改变、次正规概率乘大V，以及完整d64范围外的大V重缩放回归。常规小形状与CPU FP64全输出比较，S=16384,d=64只验证两种模式各10行，使用rtol=atol=2e-5。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m tests.test_cuda_fa2_forward --impl cuda_fa2_v5 --device cuda
```

分别使用`memcheck`、`synccheck`和`racecheck`执行16项常规测试，长序列不放入sanitizer。逐项结果见[data](data/)和manifest。注册/路由测试使用`python -m unittest tests.test_forward_registry -v`。验证不以某次sanitizer前的旧扩展结果替代最终二进制结果。

## 1.4 独立与成对计时

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.benchmark_attention --impl cuda_fa2_v5 --device cuda \
  --seq-len 16384 --head-dim 64 --no-causal --no-verify \
  --output-json docs/cuda-fa2/measurements/v5/data/benchmark-custom.json
```

独立计时为5次warmup、每批20次、5批。efficient用`--impl efficient`。全部forward开销进入计时，包括输出与scratch分配、bounds清零、Q/K打包、主路径/回退kernel启动、partial写回和combine，不在计时外缓存打包数据。CPU参考与输入生成在计时外，`--no-verify`不表示省略独立正确性验证。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.benchmark_attention_comparison --impl cuda_fa2_v5 --suite paired \
  --output-json docs/cuda-fa2/measurements/v5/data/benchmark-paired.json
```

`paired`使用seed0/1/7，每种causal模式各5轮，相同Q/K/V、交替执行先后顺序，每批5次warmup及20次测量。`--suite matrix`覆盖24形状×两种模式，每案例3轮、10次测量，输出`benchmark-matrix.json`。两种实现先做CPU FP64抽样行校验，不生成完整S×S参考。报告保留所有样本，不剔除不利结果；用户认可约40%耗时减少作为本版结束条件，严格60%参考线仍显示在图中。

## 1.5 Counters、时间线和资源

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/nvprof --profile-from-start off --csv \
  --metrics achieved_occupancy,eligible_warps_per_cycle,issue_slot_utilization,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_sync \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.profile_attention --impl cuda_fa2_v5 --no-causal --warmup 1
```

最终完整29项counter名称与各kernel结果见manifest的`profile`，原始CSV为`data/nvprof-custom-s16384-d64.csv`。辅助kernel与主体分别报告；counter replay耗时不作为性能验收数字。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/nvprof --profile-from-start off --csv --print-gpu-trace \
  --export-profile docs/cuda-fa2/measurements/v5/profiles/forward.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python \
  -m scripts.profile_attention --impl cuda_fa2_v5 --no-causal --warmup 1
```

```bash
cuobjdump --dump-resource-usage docs/cuda-fa2/measurements/v5/binaries/cuda_fa2_v5_extension.cpython-311-x86_64-linux-gnu.so
cuobjdump --dump-sass docs/cuda-fa2/measurements/v5/binaries/cuda_fa2_v5_extension.cpython-311-x86_64-linux-gnu.so
```

资源与静态指令计数集中放入manifest。完整SASS、临时resource文本、object、SQLite和编译缓存可重新生成，不进入正式归档。原始`.nvvp`、测量二进制及其SHA-256保留。

## 1.6 消融复现与归档规则

`run1/`至`run99/`保存探索；run72只做离线分析，没有GPU计时，run12是未重编译的无效对照。run100修复回退并恢复主路径，run101清理未引用源码后完成最终构建验证。每轮假设与结果见[优化设计](optimization-design.md)，每轮命令、状态、有效源码与二进制路径见manifest的`exploratory_runs`。

`scripts/run_fa2_v5_experiment.py N`在标准单线程CUDA环境中归档当前源码、强制构建、数值测试与主形状独立计时。它不能自动恢复历史源码；复现旧实验应先使用该轮快照，按manifest的路径映射恢复被去重的文件。`scripts/schedule_fa2_v5.py`只用于历史SASS消融，最终扩展使用标准编译，脚本不会在普通构建中运行。

源码快照包含当轮完整`.cu/.cuh`，公共绑定、build入口与Python测试记录工作区哈希。字节完全相同的重复快照/二进制可只存一份，并在manifest中记录路径映射；原始时长样本和失败状态不合并为虚构结果。Git提交需要用户另行授权，本版总结不执行提交。
