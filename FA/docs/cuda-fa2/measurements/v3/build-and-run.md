# 1. V3 编译、数值验证与性能采集

V3 的开发实现为 [fa2_forward.cu](../../../../cpp_cuda/fa2/v3/fa2_forward.cu)，共用构建选择 `--version v3`，运行选择 `--impl cuda_fa2_v3`，独立模块为 `src.cuda_fa2_v3_extension`。算法与修订依据见 [优化设计](optimization-design.md)，完整结果见 [perf_v3](perf_v3.md)，文件身份见 [manifest](manifest.json)。

## 1.1 文件与实验轮次

```text
cpp_cuda/fa2/v3/fa2_forward.cu       当前 V3 核心实现
cpp_cuda/fa2/build_extension.py     共用编译入口，默认仍为 V1
src/cuda_fa2_forward.py             独立模块加载、调用、SHA-256 记录
src/forward_registry.py             benchmark/profile 的版本选择
tests/test_cuda_fa2_forward.py      数值、tile、分组抵消与长序列检查
docs/cuda-fa2/measurements/v3/
├── optimization-design.md         寄存器矩阵块、共享复用和展开范围
├── build-and-run.md                复现命令
├── perf_v3.md                      最终结果、efficient 对照与消融
├── efficient-analysis.md           efficient 的资源、SASS 与源码证据
├── manifest.json                   最终代码、二进制、环境和数据身份
├── snapshot/fa2_forward.cu         最终被测核心源码
├── binaries/                      最终实际扩展
├── data/                          测量原始样本、CSV、构建与验证日志
├── profiles/forward.nvvp           无 counters 时间线原始报告
├── assets/register-matrix-tiles.svg  矩阵块与 shared 生命周期图
├── efficient-reference/           efficient 的原始时间线与 counters
└── run1/ ... run6/                探索、消融、失败试验的独立归档
```

主目录固定最终的 64 行 / CTA、32 key / tile、4×4 score / thread、QK/PV 受控展开版本。run1/run2 是小幅累加优化，run3/run6 为完全展开矩阵块，run4 为 64-key 的编译资源失败，run5 为 exp2f 退化；各自身份和结果由 manifest 记录。共用 Python/绑定/测试只引用工作区，资源、SASS 计数、counter 汇总和环境集中保存于 manifest，完整导出不追踪。run3/run6 共用 run6 的同哈希源码快照；相同 registry/memcheck 输出也通过 manifest 共用一份，各轮不同二进制与原始测量仍独立保留。

## 1.2 编译与资源控制

当前机器为 GTX 1060、sm_61，Python 环境 nanovllm、PyTorch 2.6.0+cu124、nvcc 12.8.93、g++ 13.3、driver 570.211.01。CPU threads/MAX_JOBS 为 1，FP32、`expf`、`-O2`，未启用 `--use_fast_math`。本 IDE 普通进程的 CUDA 初始化受沙箱限制，沿用用户授权的 sudo 子进程执行；终端通过正常认证输入密码，不写入脚本或归档。

以下均在 FA 目录执行。独立版本不会覆盖 V2 模块，修改源码后需要重新构建并启动新 Python 进程，导入不会自动 JIT。

```bash
cd /home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py --version v3 build_ext --inplace --force \
  --build-temp /tmp/fa2-v3-build/temp --build-lib /tmp/fa2-v3-build/lib \
  > /tmp/fa2-v3-build.log 2>&1
```

CUDA 编译版本和 PyTorch build 的 minor mismatch、g++ bounds 缺失和旧架构编译弃用提示均保留在 build.log。长时间任务可用 `tmux new -s fa2-v3`，GPU benchmark 与 profiler 顺序执行。

## 1.3 正确性与同步检查

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v3 \
  --long-sequence --output-json /tmp/fa2-v3-row-validation.json

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m unittest tests.test_forward_registry

for tool in memcheck synccheck racecheck; do
  sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/local/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
    -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v3
done
```

常规 12 项测试加长序列 1 项：全部 d=1..128、query/tile/causal 边界、max 改变、输入只读、no-grad，另增加分组边界与正负抵消。S=63/64/65 覆盖最终 CTA 尾部，S=257,d=128 覆盖最高模板多块执行。`rtol=atol=2e-5` 保持不变。长序列验证 S=16384,d=64、seed 0 的十行 × 两种 mask，共 20 个结果；JSON 记录 `verified_all_rows=false`。

单行 FP64 参考的输入副本为 \(O(Sd)\)，工作区 \(O(S+d)\)、十行计算 \(O(10Sd)\)，各行独立可并行，本次 CPU 一个线程。sanitizer 只执行常规 12 项，工具 error summary、退出码与运行范围保留在 manifest 和日志。

## 1.4 正式计时与确认实验

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark_attention --device cuda --impl cuda_fa2_v3 \
  --seq-len 16384 --head-dim 64 --warmup 5 --iterations 20 --repeats 5 \
  --threads 1 --seed 0 --no-causal --no-verify \
  --output-json /tmp/fa2-v3-benchmark-custom.json
```

脚本默认 causal，non-causal 比较必须显式 `--no-causal`。causal 与 efficient 分别改参数并保存独立 JSON；Event/wall 每组除以 20 次调用，保留五组完整样本。`--no-verify` 只避开大形状完整 CPU FP64 矩阵，数值验证先按 §1.3 完成。

V2/V1/V0 的基线直接引用归档，常规流程不复跑。如果新版本结果与历史基线的差异接近噪声，优先重复新版本确认实验并保留所有组；不能只选更快一轮。各轮条件与解释以性能文档和 manifest 为准。

## 1.5 Counters 与无 counters 时间线

29 项指标与 V2 相同，包含 DRAM、L2、shared 读写速率与 shared 每请求事务。5 warmup 后只 capture 一次 forward，多指标会 replay。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --kernels '.*fa2_forward_fp32_kernel.*' --aggregate-mode on \
  --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,eligible_warps_per_cycle,issue_slot_utilization,stall_sync,stall_memory_dependency,stall_exec_dependency,stall_other,stall_not_selected,stall_inst_fetch,stall_pipe_busy,stall_texture,stall_memory_throttle,stall_constant_memory_dependency,dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,l2_tex_read_hit_rate,dram_read_transactions,l2_read_transactions,gld_efficiency,gst_efficiency,shared_load_throughput,shared_store_throughput,shared_load_transactions_per_request,shared_store_transactions_per_request \
  --print-gpu-trace --csv --log-file /tmp/fa2-v3-counters.csv \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention --impl cuda_fa2_v3 \
  --seq-len 16384 --head-dim 64 --warmup 5 --threads 1 --seed 0 --no-causal

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --print-gpu-trace --csv --log-file /tmp/fa2-v3-timeline.csv \
  --export-profile /tmp/fa2-v3-forward.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention --impl cuda_fa2_v3 \
  --seq-len 16384 --head-dim 64 --warmup 5 --threads 1 --seed 0 --no-causal
```

正式速度比取无 profiler benchmark，replay/capture 时间不替代它。吞吐不乘独立计时来还原字节量，stall 百分比不当成墙钟时间占比。两个进程输出相同扩展身份时，只保留一份 `nvprof-process.log`；实际 CSV 中的 PID 和命令分别保留。

## 1.6 二进制依赖检查与归档

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  docs/cuda-fa2/measurements/v3/binaries/cuda_fa2_v3_extension.cpython-311-x86_64-linux-gnu.so \
  > /tmp/fa2-v3-resources.txt

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-sass \
  docs/cuda-fa2/measurements/v3/binaries/cuda_fa2_v3_extension.cpython-311-x86_64-linux-gnu.so \
  > /tmp/fa2-v3-sass.txt
```

按 R=1..4 的模板独立检查 registers、local/stack/spill、barrier、shuffle 和共享访问。主模板 R=2 需要核对 QK 的 16 个独立 score、PV 的 32 个输出 accumulator，以及受控展开后的循环跳转；单纯统计 FFMA 数不能证明独立性。各 run 必须使用自己的归档二进制。efficient 的实测 kernel 为 `fmha_cutlassF_f32_aligned_64x64_rf_sm50`，从 libtorch_cuda.so 提取时需要完整 mangled symbol，见 [分析文档](efficient-analysis.md)。

源码快照只保存核心 `.cu`，实际扩展放 `binaries/`；共用文件记录工作区路径与本次哈希。manifest 集中存环境、资源、SASS 和 counters 汇总，benchmark/FP64 JSON、原始 CSV 和日志固定真实测量；独立 counters JSON、完整派生导出与执行状态无需追踪。
