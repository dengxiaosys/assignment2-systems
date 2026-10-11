# 1. V2 构建、验证与性能采集

V2 开发代码为 [fa2_forward.cu](../../../../cpp_cuda/fa2/v2/fa2_forward.cu)，统一构建选择 `--version v2`，运行选择 `--impl cuda_fa2_v2`，模块为 `src.cuda_fa2_v2_extension`。算法与参数见 [优化设计](optimization-design.md)，本次实测见 [perf_v2](perf_v2.md)，执行身份见 [manifest](manifest.json)。以下命令均在 `FA` 目录执行。

## 1.1 目录与版本身份

```text
cpp_cuda/fa2/v2/fa2_forward.cu       V2 开发核心代码
cpp_cuda/fa2/build_extension.py     共用构建入口，默认仍为 V1
src/cuda_fa2_forward.py             版本加载、调用和 SHA-256 记录
src/forward_registry.py             benchmark/profile 的统一版本选择
tests/test_cuda_fa2_forward.py      共用数值、tile 边界和长序列验证
docs/cuda-fa2/measurements/v2/
├── optimization-design.md         首版推导与假设
├── build-and-run.md                复现步骤
├── perf_v2.md                      实际计时、资源、counters 与限制
├── manifest.json                   源码、二进制、环境与归档哈希
├── snapshot/fa2_forward.cu         只固定本次核心 CUDA 代码
├── binaries/                      实际被测扩展
├── data/                          构建、测试、计时、原始 CSV
├── profiles/forward.nvvp           原始 profiler 报告
└── assets/tile-protocol.svg        数据所有权与同步图
```

公共绑定、Python、测试、构建脚本只维护工作区的一份，snapshot 不重复复制这些文件。V0/V1 的 benchmark 与 counters 直接引用原归档，本轮不复跑。多轮独立测量应使用新的 `run1/` 等位置，避免覆盖本次证据。

## 1.2 环境与编译

本次环境为 GTX 1060、sm_61、CUDA 12.8.93、g++ 13.3、PyTorch 2.6.0+cu124。CPU 线程和编译任务均限制为 1。构建使用 `-O2 -lineinfo --ptxas-options=-v`，保留 `expf`，不开启 `--use_fast_math`；CUDA 编译版本与 PyTorch CUDA build 的 minor mismatch 警告保存在日志。

当前 IDE 沙箱拦截普通进程 CUDA 初始化时的 `/proc/.../task/.../comm` 操作，表现为 error 304；sudo 子进程初始化成功。因此本次构建、CUDA 测试、benchmark、sanitizer 和 nvprof 均通过用户已授权的 sudo 运行。普通终端不受该限制时可移除 sudo；计数器仍受 `RmProfilingAdminOnly=1` 限制。密码只通过认证输入传递，未写入脚本、数据或 manifest。

```bash
cd /home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST=6.1 MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py --version v2 build_ext --inplace \
  --build-temp /tmp/fa2-v2-build/temp --build-lib /tmp/fa2-v2-build/lib \
  > /tmp/fa2-v2-build.log 2>&1
```

`--version v2` 只生成独立 V2 模块。修改 `.cu` 后必须重建并启动新的 Python 进程；扩展导入不执行 JIT 构建。长期任务可用 `tmux new -s fa2-v2`，GPU 的 benchmark 和 profiler 按顺序执行。

## 1.3 正确性与 shared 协议

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v2 \
  --long-sequence --output-json /tmp/fa2-v2-row-validation.json

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m unittest tests.test_forward_registry
```

常规 11 项测试包含全部 d=1..128、query/维度尾部、S=31/32/33 等 key tile 边界、causal 对角 mask、跨 tile 最大值变化和输入/输出契约。长序列额外一项测试，在 S=16384,d=64、seed 0 上验证行 `0,1,3,4,7,8,31,32,8191,16383` × 两种 causal 设置，合计 20 行参考结果。

FP64 参考只物化单行 scores，CPU 输入副本空间为 \(O(Sd)\)、单行额外工作区为 \(O(S+d)\)、十行参考计算为 \(O(10Sd)\)；各行独立可并行，本次线程为 1。容差保持 `rtol=atol=2e-5`。JSON 明确记录 `verified_all_rows=false`。

```bash
for tool in memcheck synccheck racecheck; do
  sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/local/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
    -m tests.test_cuda_fa2_forward --device cuda --impl cuda_fa2_v2
done
```

racecheck 在 V2 必须关注 shared 发布与复用：无效 query warp 仍参加两个屏障，不能沿用 V1 的提前返回。工具的退出码、error summary 和实际覆盖范围保存在 data，不能以源码审查替代检查结果。

## 1.4 无 profiler 正式 benchmark

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.benchmark_attention \
  --device cuda --impl cuda_fa2_v2 --seq-len 16384 --head-dim 64 \
  --warmup 5 --iterations 20 --repeats 5 --threads 1 --seed 0 \
  --no-causal --no-verify --output-json /tmp/fa2-v2-benchmark-custom.json
```

causal 单独将 `--no-causal` 换为 `--causal`，efficient 对照单独换成 `--impl efficient`。脚本默认 causal，不能省略 mask 参数后和 non-causal 基线比较。`--no-verify` 只避免大规模全矩阵 FP64 验证，正确性已按 §1.3 独立执行。Event/wall 每组除以 20 次调用，保留五组样本；extra peak allocation 只描述新增分配，不是 IO 流量。

## 1.5 nvprof counters 与时间线

本轮在 V1 的 23 个指标上增加 `l2_write_throughput`、`gst_efficiency`、shared 读写吞吐及每请求事务数，共 29 项。旧版缺失的 L2/shared 指标保留“未采集”，不以新版本数值回填。

```bash
sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --kernels '.*fa2_forward_fp32_kernel.*' --aggregate-mode on \
  --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,eligible_warps_per_cycle,issue_slot_utilization,stall_sync,stall_memory_dependency,stall_exec_dependency,stall_other,stall_not_selected,stall_inst_fetch,stall_pipe_busy,stall_texture,stall_memory_throttle,stall_constant_memory_dependency,dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,l2_tex_read_hit_rate,dram_read_transactions,l2_read_transactions,gld_efficiency,gst_efficiency,shared_load_throughput,shared_store_throughput,shared_load_transactions_per_request,shared_store_transactions_per_request \
  --print-gpu-trace --csv --log-file /tmp/fa2-v2-counters.csv \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention --impl cuda_fa2_v2 \
  --seq-len 16384 --head-dim 64 --warmup 5 --threads 1 --seed 0 --no-causal

sudo env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda-12.8/bin/nvprof --profile-from-start off \
  --print-gpu-trace --csv --log-file /tmp/fa2-v2-timeline.csv \
  --export-profile /tmp/fa2-v2-forward.nvvp \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention --impl cuda_fa2_v2 \
  --seq-len 16384 --head-dim 64 --warmup 5 --threads 1 --seed 0 --no-causal
```

capture 只包含一次 forward，warmup、输入生成与身份哈希在范围外。多指标触发 replay；profiler 时间不替代 benchmark，吞吐不乘以独立计时还原字节量。任何 overflow 保留并排除定量比较。工具路线见 [Pascal profiling 说明](../../profiling-tools.md)。

## 1.6 资源、SASS 与归档

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  docs/cuda-fa2/measurements/v2/binaries/cuda_fa2_v2_extension.cpython-311-x86_64-linux-gnu.so > /tmp/fa2-v2-resources.txt

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/local/cuda-12.8/bin/cuobjdump --dump-sass \
  docs/cuda-fa2/measurements/v2/binaries/cuda_fa2_v2_extension.cpython-311-x86_64-linux-gnu.so > /tmp/fa2-v2-sass.txt
```

按 `DIMS_PER_LANE=1..4` 分模板核对 barrier、shared、shuffle 和 spill。ptxas 的 `used 1 barriers` 是使用的 barrier 资源槽数量，不是只执行一次屏障；本协议两个源码屏障均使用同一槽，SASS 中存在两个静态 `BAR.SYNC` 位置。静态指令位置也不是全网格动态次数。

归档核心源码和实际加载的 `.so`，并核对 benchmark/profile 输出的 SHA-256 与归档二进制一致。manifest 集中保存设备属性、模板资源和 SASS 静态计数；完整资源/SASS 导出与执行状态文件可以删除，重新生成时不纳入 Git。时间线与 counters 的进程身份相同，只保留 `data/nvprof-process.log` 一份。公共文件路径与哈希、环境、flags、全部样本和数据哈希仍由 manifest 记录；共享文件由工作区维护，性能结论逐项对照 [H1–H7](optimization-design.md#hypothesis-validation)。
