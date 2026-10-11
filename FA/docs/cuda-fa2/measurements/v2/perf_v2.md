# 1. V2 实测：K/V shared tile 与分块 softmax

V2 已完成实现、编译、CUDA 数值验证、memcheck/synccheck/racecheck、正式 benchmark 和 29 项 nvprof 指标采集。GTX 1060、FP32、\(S=16384,d=64\)、non-causal 下，Event 均值为 **143.560 ms**，相对归档 V1 的 **348.178 ms** 快 **2.425 倍**，调用时间减少 **58.768%**。causal 为 **73.508 ms**，相对归档 V1 causal 快 **2.482 倍**。本轮 efficient 为 **36.775 ms**，V2 仍慢 **3.904 倍**。

本轮验证了“共享 K/V tile + 按 tile 更新 softmax”的优化方向。主形状的 global K/V 有效逻辑读取从 128 GiB 减为 16 GiB，每行跨 key 状态更新从 16384 次减为 512 次。实际 DRAM 读事务增加，不能将上述结构变化解释为显存读取减少 8 倍。当前最大 stall 分类转为 **exec dependency 53.507%**；继续优化应关注 QK/PV 累加链与指令级并行，而不是仅增大 shared tile。

首版推导和假设见 [optimization-design](optimization-design.md)，操作命令见 [build-and-run](build-and-run.md)，实际版本身份和全部数据哈希见 [manifest](manifest.json)。测量日期 2026-10-12；V0/V1 数据直接引用历史归档，本轮未重新运行它们。

## 1.1 实现与测量身份

### 1.1.1 数据组织

[常量与 shared 布局](snapshot/fa2_forward.cu#L16-L45) 固定 `WARPS_PER_BLOCK=8`、`KEYS_PER_TILE=32`、`THREADS_PER_BLOCK=256`。一个 warp 独占一个 query，Q/Z 保持每 lane 的条带寄存器。CTA 协作将 K 转置到 `[D][33]`，V 保持 `[32][D]`；`D=32*DIMS_PER_LANE`，padding 为计算出的 `KEYS_PER_TILE+1`。

[QK 阶段](snapshot/fa2_forward.cu#L81-L96) 中，每个 lane 负责一个 key 的 score，通过 Q 分量 shuffle 和 shared K 沿 d 累加，因此 32 个 score 可以并行计算。[tile softmax 与 PV](snapshot/fa2_forward.cu#L97-L122) 用 warp max/sum all-reduce 更新状态，广播 32 个概率，再让 lane 更新自己负责的输出维度；没有跨 warp 输出归约。`expf`、FP32 和最终一次归一化保持不变。

[两个 CTA 屏障](snapshot/fa2_forward.cu#L67-L124) 分别保护 tile 发布和 shared 复用。query 尾部 warp 不提前返回，所有线程执行共同的加载循环与屏障；causal 对角 tile 按每行 `key_limit` mask，完全未来的 tile 跳过该 warp 算术。同步证明和边界说明见 [设计 §1.4](optimization-design.md#tile-synchronization)。

### 1.1.2 实际运行二进制

实际模块为 `src.cuda_fa2_v2_extension`，SHA-256 为 `69f273550022c1e699e9c12c8b226494e8cca96d4c58df1299d024692b3f2e32`。归档 [实际扩展](binaries/cuda_fa2_v2_extension.cpython-311-x86_64-linux-gnu.so) 与 benchmark、FP64 行验证、profile 进程记录的哈希一致。[核心快照](snapshot/fa2_forward.cu) 固定被测源码；[开发源码](../../../../cpp_cuda/fa2/v2/fa2_forward.cu) 后续仅补充了 V1 风格的空行分组与格式注释，未改变计算逻辑。

统一构建、公共 binding、Python 和测试引用工作区，manifest 记录其本次哈希。CPU threads/MAX_JOBS 均为 1，CUDA 12.8.93、g++ 13.3、PyTorch 2.6.0+cu124、驱动 570.211.01。当前沙箱普通进程 CUDA 初始化遇到 error 304，用户授权的 sudo 子进程成功运行；本轮 GPU 操作均在 sudo 下完成。密码未进入脚本或归档。

## 1.2 编译资源、SASS 与驻留成本

### 1.2.1 分模板资源

来源：[build.log](data/build.log) 与 [manifest 的 compiled_resources](manifest.json)。完整 cuobjdump 资源导出可从归档二进制再生成，见 [复现 §1.6](build-and-run.md#L111-L127)。

| DIMS_PER_LANE | 支持 d | registers / thread | shared / CTA | local / stack / spill |
|---:|---|---:|---:|---|
| 1 | 1–32 | 40 | 8320 B | 均为 0 |
| 2 | 33–64 | 40 | 16640 B | 均为 0 |
| 3 | 65–96 | 40 | 24960 B | 均为 0 |
| 4 | 97–128 | 32 | 33280 B | 均为 0 |

主模板相对 V1 从 23 增至 40 registers/thread，引入 16640 B shared。最高 R 的寄存器反而为 32，说明编译器的展开和调度策略不同，不能按源码数组大小线性预测 registers。四模板均无 spill，ptxas 报告 `used 1 barriers`，表示共用一个 barrier 资源槽。

### 1.2.2 指令与占用的核对

[manifest 的分模板 sass_counts](manifest.json) 记录：每个模板均有两个 `BAR.SYNC` 和两个 `MEMBAR.CTA` 静态位置。主模板有 107 个 `SHFL`、128 个 `LDS` 静态位置，对应 64 次 Q 广播、5 次 max、1 次 alpha、5 次 sum、32 次 P 广播，以及 K/PV 的 shared 消费。完整 SASS 可从归档二进制再生成；静态位置不能直接当成全网格动态次数，`MUFU` 也不能直接当成源码 `expf` 调用次数。

本设备实测资源上限为每 SM 98304 B shared、65536 个 32-bit registers、2048 threads，来源 [manifest 的 environment.device_attributes](manifest.json)。主模板仅按 shared 计算最多 \(\lfloor98304/16640\rfloor=5\) 个 CTA，按寄存器原始数量计算 \(\lfloor65536/(256\cdot40)\rfloor=6\)，线程上限为 8。shared 因而给出 5 CTA × 8 warps = 40 warps、62.5% 的资源上限；真实分配还受资源粒度等限制。nvprof achieved occupancy 为 62.4171%，与该上限接近。

这说明 V2 以驻留数量换取数据复用，不能假设同时保持 V1 的 98.5105% occupancy。优化后总时间仍显著下降，所以高 occupancy 本身不是最终目标。

## 1.3 数值、越界与同步验证

[CUDA 验证](data/cuda-validation.log) 12 项测试通过，其中常规 11 项加长序列 1 项；[registry](data/registry-validation.log) 5 项通过。覆盖全部 d=1..128、query 尾部、key tile 31/32/33 等边界、causal 对角 mask、跨 tile max 改变、uniform attention、输入只读和输出属性。FP64 容差保持 `rtol=atol=2e-5`。

[长序列行参考](data/row-validation.json) 对 S=16384,d=64、seed 0 的行 `0,1,3,4,7,8,31,32,8191,16383` × causal/non-causal 共 20 个结果通过，最大绝对误差 **2.337974×10⁻⁷**。仅验证这些行，`verified_all_rows=false`，未物化完整 \(S^2\) 参考矩阵。有限输入及 FP32 中间不溢出的数值边界沿用 [设计 §1.2](optimization-design.md#mathematical-model)。

| 工具 | 结果 | 原始依据 |
|---|---|---|
| memcheck | exit 0；0 errors | [日志](data/memcheck.log) |
| synccheck | exit 0；0 errors | [日志](data/synccheck.log) |
| racecheck | exit 0；0 errors、0 warnings、0 hazards | [日志](data/racecheck.log) |

sanitizer 执行常规 11 项，不包含长序列抽样。它们补充尾部地址和 shared 协议的检查；全员 barrier 参与及读写所有权仍由源码控制流与设计证明支持。

## 1.4 正式 benchmark 与时间线

### 1.4.1 Non-causal 主比较

口径为相同 GPU、FP32、S=16384,d=64、non-causal、TF32 off、seed 0、CPU threads 1、5 warmup、每组 20 次、5 组重复。CUDA Event/wall 每组除以调用次数，以下使用均值比均值。V0/V1 为历史结果，频率、温度、工具链和运行轮次差异仍存在。

| 实现 | Event 均值 | Event 中位数 | V1 / 本实现 | 来源 |
|---|---:|---:|---:|---|
| V0 归档 | 2922.297 ms | 2916.777 ms | 0.119 | [V0 evidence](../v0/data/evidence.json) |
| V1 归档 | 348.178 ms | 348.262 ms | 1.000 | [V1 JSON](../v1/data/benchmark-custom.json) |
| V2 本轮 | 143.560 ms | 143.373 ms | 2.425 | [V2 JSON](data/benchmark-custom.json) |
| efficient 本轮 | 36.775 ms | 36.796 ms | 9.468 | [efficient JSON](data/benchmark-efficient.json) |

V0 / V2 为 **20.356**，V2 / efficient 为 **3.904**。V2 五组 Event 均值样本为 144.356、142.445、143.373、142.830、144.799 ms，组间有约百分之一量级变化，完整精度保存在 JSON。三次本轮 benchmark 的 extra peak allocation 都为 4 MiB，等于输出存储；不表示 kernel 总共只读写 4 MiB。

### 1.4.2 Causal 独立记录

[V2 causal](data/benchmark-custom-causal.json) 均值 **73.508 ms**、中位数 **73.637 ms**。相对历史 [V1 causal](../v1/data/benchmark-custom-causal.json) 的 182.430 ms 快 **2.482 倍**；本轮没有 efficient causal 新对照，不能从 non-causal 推导它的结果。causal 与 non-causal 的工作量和 mask 路径不同，分别报告。

### 1.4.3 单 kernel capture

不带 counters 的 [nvprof 时间线](data/nvprof-timeline.csv) 捕获一个主模板 kernel：**144.913198 ms**，grid `(2048,1,1)`、block `(256,1,1)`、40 registers/thread、static shared 16.25 KiB、dynamic shared 0 B。原始 [forward.nvvp](profiles/forward.nvvp) 和 [进程身份](data/nvprof-process.log) 保留。时间线与 counters 进程输出了完全相同的扩展身份记录，归档共用一份。

时间线与独立 benchmark 的量级一致，正式速度比仍取无 profiler 的五组结果，不用一次 capture 或 counters replay 替代。

## 1.5 Counters、存储读写与剩余依赖

### 1.5.1 采集口径

本轮 nvprof 12.8.90 (21) 在 5 warmup 后捕获一次 V2 forward，`aggregate-mode on`，29 项指标触发 replay，只出现预期的 `DIMS_PER_LANE=2` 模板，没有 overflow。来源为 [原始 CSV](data/nvprof-custom-s16384-d64.csv)、[counters.json](data/counters.json)、[进程日志](data/nvprof-process.log)。V1 数值直接取历史 [counters](../v1/data/counters.json)，不是配对同轮实验。

### 1.5.2 驻留、发射与 stall

| 指标 | V1 归档 | V2 本轮 | 判读 |
|---|---:|---:|---|
| achieved occupancy | 98.5105% | 62.4171% | shared 的驻留限制与实测相符 |
| eligible warps / cycle | 5.394563 | 3.211435 | 更少驻留，加上计算依赖 |
| issue slot utilization | 71.425499% | 50.959040% | 发射利用率下降，工作量减少仍使整体更快 |
| warp execution efficiency | 54.256673% | 98.384853% | score/概率阶段各 lane 参与更多 |
| warp nonpred execution efficiency | 54.169450% | 97.737087% | 同上，不能单独解释为有用 FLOP 效率 |
| `stall_memory_dependency` | 39.899801% | 6.239920% | 新的数据组织显著改变等待组成 |
| `stall_exec_dependency` | 21.597141% | 53.507239% | 当前最大的已分类等待 |
| `stall_sync` | 0.000000% | 7.668617% | tile 共享协议引入 CTA 汇合 |
| `stall_other` | 24.746867% | 24.289828% | 仍未细分，不能归入已知原因 |
| `stall_inst_fetch` | 8.727479% | 3.610647% | 本轮未显示增加 |
| `stall_not_selected` | 4.215677% | 3.093487% | 驻留与就绪 warp 同时改变 |

上述比例是停顿原因分布，不能乘以 143.560 ms 得到绝对等待时间。memory 比例下降不能证明其绝对周期下降了某个固定倍数，exec 比例上升也不能证明它比 V1 更耗时；缺少各分类的独立绝对周期分解。

### 1.5.3 DRAM、L2 与 shared 读写速率

按每行统一单位，MB/GB 为十进制字节，KiB/GiB 用于二进制容量。数值直接来自相应 profiler pass，未通过逻辑字节和 Event 时间推算。

| 层级与操作 | 单位 | V1 归档 | V2 本轮 |
|---|---|---:|---:|
| DRAM read | GB/s | 0.603042352 | 2.251748 |
| DRAM write | MB/s | 11.638323 | 28.126977 |
| L2 read | GB/s | 40.276353 | 69.605342 |
| L2 write | MB/s | 未采集 | 28.258494 |
| shared load | GB/s | 未采集 | 888.284720 |
| shared store | GB/s | 未采集 | 111.035590 |

V2 shared 读吞吐约为写吞吐的 8 倍，与 8 query warp 消费一次 CTA tile 的结构一致；这只能作为结构的补充证据，不能将不同采集 pass 的吞吐乘以独立 benchmark 时间还原流量。DRAM 与 L2 写入速率接近，不表示输出之外的层级写入被完全穷尽；源码仍只在 tile 循环结束后写回 O。

### 1.5.4 事务、命中率与 bank 检查

| 指标 | V1 归档 | V2 本轮 |
|---|---:|---:|
| DRAM read transactions | 6966350 | 10887482 |
| L2 read transactions | 476439296 | 336550499 |
| L2 texture read hit rate | 88.602554% | 94.709465% |
| global load efficiency | 100.000000% | 100.000000% |
| global store efficiency | 未采集 | 100.000000% |
| shared load transactions / request | 未采集 | 1.000000 |
| shared store transactions / request | 未采集 | 1.000000 |

DRAM 读事务为 V1 的 **1.563 倍**，L2 读事务为 V1 的约 **1/1.416**。global K/V 逻辑加载虽减少 8 倍，V1 原本已通过缓存实现大量隐式复用；V2 改变加载方式、CTA 调度、缓存与驻留，实际存储层事务不遵循同一倍数。当前数据不足以唯一确定 DRAM 事务增加的细分原因，不能将其全部归因于某一缓存层或 replay。

shared transactions/request 读写均为 1，支持 padding 布局在本轮主模板中未出现额外的 bank 事务。它不保证所有 d 或以后变体都无冲突；布局公式、SASS 与测试结果应一起审查。V1 没有 shared 数据工作区，但其 shared 指标未采集，表中保持缺失状态。

## 1.6 假设核验与下一优化方向

| 假设 | 结果 | 证据 |
|---|---|---|
| H1：global K/V 逻辑加载少 8 倍 | 结构成立；硬件流量不是 8 倍变化 | [协作加载](snapshot/fa2_forward.cu#L67-L78)、§1.5.4 |
| H2：每行状态更新少 32 倍 | 成立，16384 → 512 | [tile 更新](snapshot/fa2_forward.cu#L97-L122) |
| H3：每 tile 两屏障，尾部全员参与 | 源码与 SASS 成立，检查工具通过 | [manifest 的 sass_counts](manifest.json)、[racecheck](data/racecheck.log) |
| H4：主模板 shared 16640 B、无 spill | 成立；40 registers/thread | [manifest 的 compiled_resources](manifest.json)、[build.log](data/build.log) |
| H5：延迟低于 V1 | 成立；2.425 倍 | [benchmark](data/benchmark-custom.json) |
| H6：等待或发射行为改善 | memory 原因占比降低；issue/eligible 下降，exec 成为最大分类 | [counters](data/counters.json) |
| H7：数值、causal 和 tail 成立 | 已测用例通过，长序列只抽样 | [测试](data/cuda-validation.log)、[行参考](data/row-validation.json) |

V2 后最值得研究的是 [QK 的 `s += q_value * k_tile` 链](snapshot/fa2_forward.cu#L84-L95) 与 [PV 的 `z += p*V` 链](snapshot/fa2_forward.cu#L108-L122)。主模板每 lane 沿 64 个维度求一个 score，PV 又依次消费 32 个概率；这与 exec dependency 高占比相容，但 profiler 没有把 53.507% 单独分配给两条链或 `expf`。

下一实验可优先尝试多个独立 QK partial accumulator，分散 FMA 依赖后再合并，或寄存器预取下一组 shared 操作，同时检查新增寄存器是否进一步降低驻留。这些变化应进入新的独立版本，不覆盖 V2 快照。盲目扩大 key tile 会增加 shared、降低驻留并加长 PV 链；当前 global load 已合并、shared 每请求事务为 1，因此继续处理数据依赖比单纯修正 bank 或增大 tile 更有直接证据。

所有性能结论限于本次主形状与设备。V2 属于组合优化，历史 V1 与本轮条件不同；无全形状性能保证、无全行 FP64 证明、无非有限输入扩展语义。原始结果及构建/运行对象均由 [manifest](manifest.json) 固定。
