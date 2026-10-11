# 1. V1 实测：warp 内归约与四行 / CTA

V1 已完成实现、编译、CUDA 正确性验证、正式 benchmark 和 nvprof 采集。GTX 1060、FP32、\(S=16384,d=64\)、non-causal 下，V1 的单次 forward CUDA Event 均值为 **348.178 ms**，直接引用 V0 已归档基线 **2922.297 ms**，速度比 **8.393 倍**。循环内 CTA barrier 已删除，`stall_sync` 实测为 **0%**。相比本次 efficient 的 **37.911 ms**，V1 仍慢 **9.184 倍**，逐 key 状态递推和逻辑 K/V 重读继续存在。

存储吞吐实测为：**DRAM 读取 603.042352 MB/s、DRAM 写入 11.638323 MB/s、L2 读取 40.276353 GB/s**；完整 V0/V1 对比见 [显存与 L2 读写速率](#memory-throughput)。L2 写入速率未在本轮采集。

代码与复现命令见 [编译执行文档](build-and-run.md)，优化前的数学模型与 H1–H8 见 [设计文档](optimization-design.md#performance-hypotheses)，本次源码、二进制和环境身份见 [manifest](manifest.json)。测量日期为 2026-10-11；目录和文件名按版本与用途组织。

本文的代码说明采用最新开发源码命名：索引以 `i` 结尾，全局常量和整数模板参数使用全大写。性能、资源与 SASS 证据仍绑定本次测量快照；快照和 profiler 输出保留采集时的名称，不随开发源码重命名。

## 1.1 实现范围、命名与版本身份

### 1.1.1 本次实施的优化

[V1 开发源码](../../../../cpp_cuda/fa2/v1/fa2_forward.cu#L16-L48) 使用 `WARP_SIZE=32`、`WARPS_PER_BLOCK=4`、`THREADS_PER_BLOCK=WARP_SIZE * WARPS_PER_BLOCK`，一个 warp 处理一个 query。`warpi` 是 block 内 warp 索引，`lanei` 是 warp 内 lane 索引，query 行为 `qi = blockIdx.x * WARPS_PER_BLOCK + warpi`；槽位 `ri` 对应维度 `dimi = ri * WARP_SIZE + lanei`。每 lane 的 `q_dim[DIMS_PER_LANE]` 与 `z[DIMS_PER_LANE]` 保存自己负责的 Q 与输出分量，key 行索引为 `ki`。`m/l/z/alpha/p_tilde/o` 保持算法命名，`l` 对应分母 ell，`z` 对应未归一化输出分子。

[warp_sum](../../../../cpp_cuda/fa2/v1/fa2_forward.cu#L22-L27) 使用五级 `__shfl_down_sync` 汇集局部点积 `reduction`，完整结果仅由 `lanei == 0` 的线程使用。该线程计算 score `s`、更新 `m_new/m/l` 后，在条件体之外通过 `FULL_WARP_MASK` 广播 `alpha/p_tilde`。循环结束时通过一次 shuffle 得到最终分母 `l_final`，各线程写自己拥有的输出维度。query 行尾按整 warp 返回，维度尾部只屏蔽数据加载与更新，完整 32-lane mask 保持有效；对应 [权重与输出更新](../../../../cpp_cuda/fa2/v1/fa2_forward.cu#L54-L95)。

最新 [host 分派](../../../../cpp_cuda/fa2/v1/fa2_forward.cu#L124-L150) 先计算运行时槽位数 `dims_per_lane = 1 + (head_dim - 1) / WARP_SIZE`，再调用 `FORWARD_LAUNCHERS[dims_per_lane - 1]`，选择编译期参数 `DIMS_PER_LANE` 为 1、2、3、4 的实例。主配置 `head_dim=64` 对应 `DIMS_PER_LANE=2`，输入上限由 `MAX_HEAD_DIM=128` 检查。归档测量时的源码使用区间分派，见 [测量快照](snapshot/fa2_forward.cu#L126-L149)；查表是后续 host 代码整理，本文的计时与资源仍来自归档版本。两种分派覆盖相同维度范围，kernel 保持 `expf`、FP32 和逐 key 状态更新，未增加 K/V shared tile 或快速指数。

### 1.1.2 实际运行对象与父版本

V1 实际加载 `src.cuda_fa2_v1_extension`，SHA-256 为 `078da892a4215b62df43f75642e84e1e107529f1ee370089326494f247ccaa47`。归档 [实际扩展](binaries/cuda_fa2_v1_extension.cpython-311-x86_64-linux-gnu.so) 与本次运行 `.so` 哈希一致。`snapshot/` 只保留核心 `fa2_forward.cu`，公共绑定、统一编译脚本与 Python 入口引用工作区；[manifest](manifest.json) 保存构建参数、日志和版本身份。

V0 的计时、源码和 counters 直接引用 [V0 manifest](../v0/manifest.json)、[V0 核心快照](../v0/snapshot/fa2_forward.cu) 与 [原始记录](../v0/data/evidence.json)，不在 V1 重复复制。V0 实际测量二进制哈希为 `29f397f8fa5bad4d2bd7f4f3ba54be87cee2d8bb5eaa4a505db283f60f957d6a`。初始化 V1 时已经执行过的一次 V0 复跑继续保留为补充证据，后续常规优化不再要求复跑 V0。

V1 使用 CUDA `12.8.93`、g++ `13.3.0` 重新编译；V0 历史完整编译工具链没有记录。8.393 倍是相同计时口径下新版本结果与历史基线的比值，源码结构和 SASS 支持其优化方向；不同采集轮次的运行条件差异仍需说明，该结果不将全部收益严格归结为单一 barrier 成本。

## 1.2 编译资源与机器指令

### 1.2.1 四个模板的资源

来源：[完整编译日志](data/build.log)、[cuobjdump 资源统计](data/resources.txt) 与 [结构化资源](data/kernel-resources.json)。

| `DIMS_PER_LANE` | `head_dim` 范围 | registers / thread | shared / CTA | local / thread | stack |
|---|---|---:|---:|---:|---:|
| 1 | 1–32 | 21 | 0 B | 0 B | 0 B |
| 2 | 33–64 | 23 | 0 B | 0 B | 0 B |
| 3 | 65–96 | 23 | 0 B | 0 B | 0 B |
| 4 | 97–128 | 27 | 0 B | 0 B | 0 B |

ptxas 对四个模板均报告 `used 0 barriers`，spill stores/loads 均为 0 B。主配置的寄存器数由 V0 的 17 增至 23，但没有 spill；`q_dim/z` 多槽位被标量化为寄存器状态，shared reduction 与权重工作区从 V0 的 524 B 变为 0 B。

### 1.2.2 屏障与 shuffle 的确认

[V1 SASS](data/sass.txt) 中四个模板都没有 `BAR.SYNC` 或 `MEMBAR.CTA`。主配置 `ILi2E` 实例有 8 个静态 `SHFL` 位置，与五级归约、两次权重广播和循环后的分母广播对应。每 query-key 对源码层 collective 数为 7，每行最终另有 1 次广播；静态位置与全网格动态调用数属于不同计数。

`DIMS_PER_LANE=1` 的编译器代码包含 22 个静态 `SHFL` 位置，体现该实例的循环展开；因此不能要求所有模板都只有 8 条静态 shuffle，也不能把 22 当作每 key 固定通信次数。无 CTA barrier 的结论由源码、ptxas 与分模板反汇编共同支持。

对于 non-causal 主形状，query-key 对仍为 \(S^2=268435456\)。V0 的 \(10S^2+S=2684370944\) 个全网格 CTA barrier 动态实例被删除；V1 源码层的 warp collective 调用总数为 \(7S^2+S=1879064576\)。这些是结构计数，不能乘一个固定指令延迟推导墙钟时间。

## 1.3 正确性与检查工具结果

### 1.3.1 小形状数值与接口

[CUDA 验证](data/cuda-validation.log) 9 项测试通过，包含 \(d=1,\ldots,128\) 的全部合法维度、query 尾部、条带边界、不同 seed、causal/non-causal、uniform attention、未来位置隔离和最大值多次改变，同时保留输入副本、输出 shape/dtype/device 与 no-grad 属性检查。[Registry 测试](data/registry-validation.log) 5 项通过，核对 V1 解析与支持检查。

CPU FP64 数学参考仍使用 `rtol=atol=2e-5`；重新排列点积求和没有导致放宽阈值。FP64 正确性结果针对测试输入，未定义非有限输入和 FP32 中间溢出的扩展语义，数值定义域见 [设计 §1.2.2](optimization-design.md#mathematical-model)。

### 1.3.2 主序列形状抽样

[长序列参考](data/row-validation.json) 对 \(S=16384,d=64\)、seed 0 的 query 行 `0,1,3,4,8191,16383` 分别执行 causal/non-causal，合计 12 个 FP64 对照结果通过。最大绝对误差为 **1.932931×10⁻⁷**。逐行参考不物化全 \(S^2\) 矩阵；结果仅覆盖这些输出行，JSON 明确记录 `verified_all_rows=false`。

### 1.3.3 越界与同步

同一 CUDA 测试套件在 [memcheck](data/memcheck.log) 和 [synccheck](data/synccheck.log) 下均通过，工具退出码为 0，`ERROR SUMMARY: 0 errors`。这为尾部地址与 warp primitive 参与提供额外检查证据；无跨 warp 写读的正确性仍以 [数据所有权与同步证明](optimization-design.md#correctness) 为依据。

## 1.4 正式 benchmark 与单 kernel 时间线

### 1.4.1 相同 non-causal 口径

主比较使用 V0 历史归档与 V1 本轮结果，均为 GTX 1060、FP32、\(S=16384,d=64\)、non-causal、TF32 关闭、seed 0、CPU threads 1、5 次 warmup、每组 20 次 forward、共 5 组。Event/wall 均为每次调用均值；V0 样本直接在原始 evidence 中读取，V1 样本保存在本版 JSON。

| 实现 | Event 均值 | Event 中位数 | 额外峰值分配 | V0 / 本实现 | 原始结果 |
|---|---:|---:|---:|---:|---|
| V0 已归档基线 | 2922.297 ms | 2916.777 ms | 4 MiB | 1.000 | [V0 evidence：cuda_fa2 记录](../v0/data/evidence.json) |
| V1 warp-query | 348.178 ms | 348.262 ms | 4 MiB | 8.393 | [JSON](data/benchmark-custom.json) |
| PyTorch efficient，本轮对照 | 37.911 ms | 37.952 ms | 4 MiB | 77.083 | [JSON](data/benchmark-efficient.json) |

V1 相比 V0 原基线的单次 forward 时间减少 **88.085%**，V1 / 本轮 efficient 为 9.184。常规版本演进继续复用这份 V0 基线，只执行新版本测量；峰值 4 MiB 对应额外输出存储，不能解读为只读写了 4 MiB。

补充记录：[初始化 V1 时的 V0 复跑](data/benchmark-v0-recheck.json) 均值为 2924.573 ms，与原基线接近，按该复跑计算的速度比为 8.400。该记录已实际产生，因此保留但不作为后续必做步骤或主分母。历史基线与新测量存在频率、温度和运行轮次差异，不声称亚百分比精度。

### 1.4.2 Causal 单独记录

V1 causal 的 Event 均值为 **182.430 ms**，中位数 **182.514 ms**，其余计时参数相同，见 [causal JSON](data/benchmark-custom-causal.json)。本轮没有正式复跑 V0 causal，因而不报告新的 causal 版本速度比；也不将 causal 与 non-causal 混算。

### 1.4.3 无 counters 的一次 capture

[nvprof 时间线](data/nvprof-timeline.csv) 捕获一次主形状 V1 forward：单 kernel 时长 **348.777 ms**，grid `(4096,1,1)`，block `(128,1,1)`，23 registers/thread，static/dynamic shared 均为 0 B。kernel 名按 profiler 原文为 `fa2_forward_fp32_kernel<int=2>`，对应 `DIMS_PER_LANE=2`；launch 配置与四行/CTA 的理论值一致。原始 [nvvp 报告](profiles/forward.nvvp) 与 [进程身份日志](data/nvprof-timeline-process.log) 一并保留。

时间线与独立 benchmark 的 348.178 ms 量级一致；正式速度比仍取无 profiler 的五组 benchmark，计数器 replay 的时长不作为替代。

## 1.5 nvprof 指标变化与剩余瓶颈

### 1.5.1 采集方式与口径

使用 `nvprof 12.8.90 (21)`，相同主形状与 non-causal，5 次 warmup 后用 `cudaProfilerStart/Stop` 捕获一次 forward，23 项指标触发 kernel replay。`--aggregate-mode on` 为跨实例聚合，kernel filter 位于 metrics 之前；本轮 CSV 仅含一个预期模板实例，所有指标返回数值，没有 `<OVERFLOW>`。

V1 指标来源为 [原始 counters CSV](data/nvprof-custom-s16384-d64.csv)、[结构化 counters](data/counters.json) 和 [进程日志](data/nvprof-process.log)。下表 V0 为历史 [V0 counters](../v0/data/counters.json)，不是本轮重新采集；同一 GPU 与工具版本提供口径参照，但不同采集轮次的缓存、频率和 replay 差异仍需保留。

### 1.5.2 驻留、发射与 stall

| 指标 | V0 历史采集 | V1 本轮采集 | 判读 |
|---|---:|---:|---|
| achieved occupancy | 99.7961% | 98.5105% | 驻留略降，性能仍显著提升 |
| eligible warps / active cycle | 3.110919 | 5.394563 | 可发射 warp 增加 |
| issue slot utilization | 54.201824% | 71.425499% | 发射利用率提高 |
| `stall_sync` | 45.266341% | 0.000000% | 与 CTA barrier 消失一致 |
| `stall_exec_dependency` | 20.619948% | 21.597141% | 寄存器归约、指数和逐 key 递推仍存在 |
| `stall_memory_dependency` | 9.338987% | 39.899801% | 成为最大分类原因，值得继续分析数据请求与延迟 |
| `stall_other` | 18.159877% | 24.746867% | 本轮没有进一步细分，不能并入已知瓶颈 |
| `stall_inst_fetch` | 5.001215% | 8.727479% | 相对原因组成改变 |
| `stall_not_selected` | 1.530019% | 4.215677% | 就绪 warp 的选择竞争增加 |
| `sm_efficiency` | `<OVERFLOW>` | 99.999383% | V0 无有效值，不计算版本差值 |

删除跨 warp 汇合后，就绪工作与 issue utilization 提高，这与相对 V0 原基线 8.393 倍的延迟改善相互支持。memory dependency 的 39.900% 表示停顿原因分布，不是 348.178 ms 的 39.900%；其相对比例上升也不等同于绝对等待时间变长。这里没有每项原因的独立绝对周期分解，不能把新旧 stall 比例乘以独立 benchmark 时间进行定量归因。

### 1.5.3 Active lane 指标的解释

| 指标 | V0 | V1 |
|---|---:|---:|
| warp execution efficiency | 92.145628% | 54.256673% |
| warp nonpred execution efficiency | 67.779263% | 54.169450% |

这些统计下降与“有效 QK/PV 维度分配改善”并不矛盾。V1 有更多独立 query warp；每个 warp 的 `expf` 与 `m/l` 递推只由 `lanei == 0` 的线程执行，其余 lane 的 active/predicated 行为降低平均值。V0 大量 barrier、控制和 shared 操作可以同时让很多 lane active，却没有承担更多 attention 乘加。上述指标统计执行指令中的线程活动，不能单独作为有用运算占比，也不能因此判定 V1 更慢。应结合总时间、SASS 与 issue 指标判读。

<a id="memory-throughput"></a>

### 1.5.4 显存与 L2 读写速率

这里的速率是 nvprof 在相应 profiling pass 中记录的实际吞吐，单位为字节/秒。数据直接来自 [V1 原始 CSV](data/nvprof-custom-s16384-d64.csv) 和 [V0 原始 CSV](../v0/data/nvprof-custom-s16384-d64.csv)，不通过逻辑加载量与独立 benchmark 时长推算。下表按每项指标统一两版单位，保留 CSV 的原始数值精度。

| 存储层级与操作 | nvprof 指标 | 单位 | V0 历史采集 | V1 本轮采集 |
|---|---|---|---:|---:|
| DRAM 读取 | `dram_read_throughput` | MB/s | 1588.894000 | 603.042352 |
| DRAM 写入 | `dram_write_throughput` | MB/s | 2.846625 | 11.638323 |
| L2 读取 | `l2_read_throughput` | GB/s | 22.637489 | 40.276353 |
| L2 写入 | `l2_write_throughput` | GB/s | 未采集 | 未采集 |

V0 DRAM 读取原始单位为 GB/s，上表按 `1 GB/s = 1000 MB/s` 转换；V1 的 603.042352 MB/s 对应 0.603042352 GB/s。MB/GB 是十进制字节单位，不与 MiB/GiB 混用。“未采集”不表示写入吞吐为零；本轮保存的 23 项指标不包含 L2 写入吞吐，不能从其他指标补算。

DRAM 指标描述到达设备显存的读写流量；L2 指标描述缓存层处理的读取流量。global load 命中缓存时可以不触达 DRAM，所以两层速率不同是正常现象，不能将 L2 的 40.276353 GB/s 当成显存带宽。当前数据也没有单独的 global load/store 请求吞吐或 shared 读写吞吐；V1 没有 shared 通信工作区，依据为 [资源记录中的 `SHARED:0`](data/resources.txt)，而非一项采集到的 shared 吞吐零值。指标口径见 [V0 的吞吐定义与判读](../v0/nvprof-readable.md)。

V1 的 DRAM 写入吞吐高于 V0，结合源码可知输出仍只在 key 循环结束后写回 `o`，并没有因此增加逐 key 的输出写入。速率同时受实际流量与采集执行时间影响，不能仅凭速率上升判断写入量增加。DRAM 读取速率下降而 L2 读取速率上升，需结合下面的事务和命中率判断。

### 1.5.5 逻辑读取与 cache 事务

| 指标 | V0 历史采集 | V1 本轮采集 |
|---|---:|---:|
| DRAM read transactions | 155720747 | 6966350 |
| L2 read transactions | 2218604078 | 476439296 |
| L2 texture read hit rate | 86.939724% | 88.602554% |
| global load efficiency | 100.000000% | 100.000000% |

本轮 V1 的 DRAM/L2 读事务分别为历史 V0 的约 1/22.353 与 1/4.657，但 [源码 K/V 加载](snapshot/fa2_forward.cu#L54-L85) 仍逐 query-key 执行，主配置逻辑 K/V 读取保持 128 GiB。新 warp 分组与更短运行改变了时间局部性和缓存访问；没有显式 K/V tile，不能把事务下降表述为已完成 shared tiling 或承诺其他形状同样下降。

DRAM 读取吞吐实际更低，而总延迟更短，这进一步说明本阶段收益不能归结为显存带宽饱和得到缓解。L2 事务数减少且吞吐上升体现不同采集结果，但吞吐对应各自 replay pass，不能用它乘以独立 benchmark 时长还原“同一次 forward 的实际字节数”。后续优化仍应同时分析逻辑加载组织、依赖和 cache 行为。

## 1.6 优化前假设的逐项核验

| 假设 | 本轮结果 | 证据 |
|---|---|---|
| H1：CTA barrier 为零 | 成立；四个模板无 `BAR.SYNC/MEMBAR.CTA` | [SASS](data/sass.txt)、[build](data/build.log) |
| H2：shared reduction / 权重工作区消失 | 成立；shared 0 B，无 spill | [resources](data/resources.txt) |
| H3：grid=4096、block=128、四行/CTA | 成立 | [源码映射](snapshot/fa2_forward.cu#L33-L38)、[时间线](data/nvprof-timeline.csv) |
| H4：sync 来源大幅减少 | 成立；本轮 `stall_sync=0%` | [CSV](data/nvprof-custom-s16384-d64.csv) |
| H5：eligible / issue 改善 | 两项均改善；occupancy 略降 | [counters](data/counters.json) |
| H6：正式延迟下降 | 成立；V0 原基线 / V1 = 8.393 | [V0 evidence](../v0/data/evidence.json)、[V1 JSON](data/benchmark-custom.json) |
| H7：逻辑 K/V 读取与逐行递推轮数不变 | 成立；硬件事务变化单独记录 | [K/V 循环](snapshot/fa2_forward.cu#L54-L85) |
| H8：数值与接口条件成立 | 测试输入全部通过；主序列仅抽样行 | [CUDA 测试](data/cuda-validation.log)、[FP64 行参考](data/row-validation.json) |

此轮证明“warp 内归约与 query 分区”在当前主形状上有效。下一阶段可依据 memory dependency、逐 key 递推和仍然重复的逻辑加载讨论 K/V tile 与 tile softmax；改变同步协议和数值顺序时，需重新设定优化前结构预期并使用新的版本目录测量。

## 1.7 证据范围与复现入口

所有原始样本、CSV、编译资源、测试与二进制哈希见 [manifest](manifest.json)，操作步骤见 [build-and-run](build-and-run.md)。外部指标定义沿用 [V0 nvprof 指标文档](../v0/nvprof-readable.md#stall-metrics) 与其 NVIDIA 来源；本轮具体数值均来自本地真实运行。

计时结论限于记录的 GTX 1060、FP32 主形状。V1 只做 forward、支持 \(1\le d\le128\)，非有限输入语义未扩展，主序列 FP64 只检查六行 × 两种 causal 设置。旧 V0 工具链身份缺失、历史基线与新结果的运行轮次差异、历史与本轮 counters 的采集差异均已保留，结果没有被解释为普适速度保证。
