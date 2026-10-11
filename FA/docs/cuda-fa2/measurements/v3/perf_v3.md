# 1. V3 实测：寄存器矩阵块、共享复用与受控展开

V3 已完成实际实现、编译、13 项 CUDA 测试、5 项 registry、memcheck/synccheck/racecheck、正式 benchmark 和 29 项 nvprof 指标采集。GTX 1060、FP32、S=16384,d=64、non-causal 下，Event 均值 **47.459 ms**，相对归档 V2 的 **143.560 ms 快 3.025 倍**，调用时间减少 **66.941%**。causal 为 **25.760 ms**，相对 V2 的 73.508 ms 快 **2.854 倍**。本轮 efficient 为 **36.171 ms**，V3 还慢 **1.312 倍**。

主要有效变化是 **64 行 / CTA、每线程 4×4 score 与多行输出的寄存器微矩阵、K/PV shared 生命周期复用，以及限制矩阵循环展开范围**。最初四路 partial 只有小幅变化，直接 exp2f 替换还退化；全部试验保留在 run1..run6。完整设计见 [optimization-design](optimization-design.md)，运行命令见 [build-and-run](build-and-run.md)，efficient 证据见 [efficient-analysis](efficient-analysis.md)，实际代码、二进制与数据哈希见 [manifest](manifest.json)。

测量日期为 2026-10-12，CPU/MAX_JOBS=1，GPU 实验顺序执行，未改频率或功耗设置。V2/V1/V0 基线直接引用归档，没有常规复跑。

## 1.1 最终实现与测量对象

### 1.1.1 代码结构

[常量与 8-lane row reduction](snapshot/fa2_forward.cu#L16-L44) 固定四个 warp，每 warp 16 行，每 CTA 64 行、32 key / tile。[Q/Z 所有权](snapshot/fa2_forward.cu#L52-L80) 令每线程拥有四行的 Q/Z 分量，d=64 时每行八个维度槽位。QK 每线程 16 个 scores，PV 每线程 32 个 Z 分量，单个操作数服务多个乘加。

[QK](snapshot/fa2_forward.cu#L96-L123) 保留 8 次维度小循环，展开寄存器槽位和 4×4 外积；[PV](snapshot/fa2_forward.cu#L176-L198) 保留 32 次 key 循环，展开四行×维度的外积。`q_dim` 的寄存器数组索引保持编译期固定，避免简单关闭整个 d 循环展开导致动态数组访问。数值仍为 FP32/expf，没有 fast-math、Tensor Core 或精度放宽。

[shared union 与四屏障](snapshot/fa2_forward.cu#L52-L198) 先存 K，全部读者结束后覆盖为 P/V，再等待 PV 消费结束。主模板 static shared=18432 B，union 按阶段计算峰值，而不是同时分配全部矩阵。causal、query/key/d 尾部与全员 barrier 参与证明见 [设计](optimization-design.md#correctness)。

### 1.1.2 二进制身份

实际模块为 `src.cuda_fa2_v3_extension`，归档 [被测扩展](binaries/cuda_fa2_v3_extension.cpython-311-x86_64-linux-gnu.so) SHA-256：

```text
686ee2a7b5e2a04906728bb22bafb4faa782f70ea96a2e40ce3db040ff4525f3
```

benchmark、FP64 行验证与 profiler 输出的扩展哈希一致。最终 [源码快照](snapshot/fa2_forward.cu) SHA-256 `96537becdf02cff3f624a62b3d6cea395eb8793766e09ff0bc16b5215d14bc27`，与开发源码在归档时相同。共用 binding、Python、构建、测试仅记录工作区路径与哈希，不复制。

## 1.2 探索、消融与方案选择

### 1.2.1 全部真实试验

| 记录 | 实现 | non-causal Event 均值 | 主模板 registers | 判定 |
|---|---|---:|---:|---|
| [run1](run1/data/benchmark-custom.json) | QK/PV 四路 partial，保留维度谓词 | 141.224 ms | 48 | 相对 V2 只少 1.63%，没有显著收益 |
| [run2](run2/data/benchmark-custom.json) | 四路 partial，成组加载与补齐零计算 | 138.068 ms | 48 | 只少 3.83%；[确认轮](run2/data/benchmark-confirmation.json) 138.134 ms |
| [run3](run3/data/benchmark-custom.json) | 64 行 / CTA 的完全展开矩阵块 | 83.615 ms | 168 | 明显改善，完整数值验证通过 |
| [run4 编译](run4/data/build.log) | 64 key 矩阵块 | 无结果 | 主模板 168 | d=128 shared=51200 B，编译未成功，未测量 |
| [run5](run5/data/benchmark-custom.json) | 完全展开矩阵块 + exp2f/base-2 scale | 118.413 ms | 178 | 资源/驻留退化，未选用 |
| [run6](run6/data/benchmark-custom.json) | 恢复 expf，完全展开矩阵块复测 | 83.954 ms | 168 | counters 指向 instruction-fetch 瓶颈 |
| [最终 V3](data/benchmark-custom.json) | 矩阵块 + QK/PV 受控展开 | **47.459 ms** | **128** | 选用，全部验证与采集成功 |

run3/run6 的核心源码相同，归档共用 [run6 源码快照](run6/snapshot/fa2_forward.cu)；重新编译后二进制哈希不同，两份实际 binary 分别保留。lineinfo/文件时序等构建因素没有逐字节稳定保证。run4 没有成功二进制，不能把其他轮结果记在该源码下。字节相同的 registry 与 memcheck 输出只保留一份，各轮在 manifest 中引用；每轮范围、状态和哈希仍单独记录。

### 1.2.2 受控展开的具体证据

| 主模板 | 完全展开 run6 | 最终 V3 |
|---|---:|---:|
| 静态 FFMA 位置 | 2304 | 416 |
| 静态 SHFL 位置 | 280 | 56 |
| registers/thread | 168 | 128 |
| occupancy | 17.9800% | 23.4428% |
| instruction-fetch stall | 65.072291% | 9.165196% |
| issue slot utilization | 27.495267% | 68.346762% |
| Event 均值 | 83.954 ms | 47.459 ms |

两版有效 attention 乘加数量与微矩阵所有权相同，静态指令减少来自保留循环；动态乘加并未减少到 416/2304。指令体积、寄存器与调度同时改变，不能把全部 43.470% 的 run6→最终时间减少单独归因于某一项。但源码、SASS、counters 和计时一致支持限制过度展开这一优化方向。

### 1.2.3 为什么没有选择 exp2f

run5 的 `exp2f` 保持 FP64 验收，但寄存器从 168 增至 178，occupancy 为 12.3078%，相比 expf 完全展开版约 18% 更低。Pascal 的寄存器分配粒度使三个四-warp CTA 放不下更高寄存器档，低驻留与其较慢计时相容。efficient 自身可以使用 exp2f，却有不同 fragment、调度和 launch bounds；不能按函数名称推断替换必然有利。

## 1.3 编译资源与数值检查

### 1.3.1 模板资源

来源：[build.log](data/build.log) 与 [manifest 的 compiled_resources/sass_counts](manifest.json)，完整导出从归档二进制再生成。

| DIMS_PER_LANE | d 范围 | registers/thread | shared/CTA | local / stack / spill |
|---:|---|---:|---:|---|
| 1 | 1–32 | 96 | 14336 B | 均为 0 |
| 2 | 33–64 | 128 | 18432 B | 均为 0 |
| 3 | 65–96 | 168 | 22528 B | 均为 0 |
| 4 | 97–128 | 200 | 26624 B | 均为 0 |

四模板各有四个静态 BAR.SYNC/MEMBAR.CTA 位置；ptxas 的 `used 1 barriers` 是一个 barrier 资源槽反复使用。主模板仅按 registers 原始数量最多四个 CTA，16 warps/SM、25% 上限；实测 23.4428% 还受尾部调度、活动周期等影响。寄存器增加并未导致 spill，低于 V2 的 occupancy 也没有阻止更高吞吐。

### 1.3.2 正确性与工具

[CUDA 测试](data/cuda-validation.log) 13 项通过，含常规 12 项和长序列 1 项；[registry](data/registry-validation.log) 5 项通过。覆盖全部 d=1..128、S=63/64/65 等 CTA/tile 边界、max 变化、causal 未来隔离、正负抵消、输入不改写、no-grad 与输出属性。FP64 容差保持 `rtol=atol=2e-5`。

[长序列抽样](data/row-validation.json) 在 S=16384,d=64、seed0 检查十行×两种 mask，共 20 个参考结果，最大绝对误差 **2.337974×10⁻⁷**。没有生成完整 \(S^2\) FP64 参考，`verified_all_rows=false`。

| 工具 | 常规 12 项结果 | 原始日志 |
|---|---|---|
| memcheck | exit0、0 errors | [日志](data/memcheck.log) |
| synccheck | exit0、0 errors | [日志](data/synccheck.log) |
| racecheck | exit0、0 errors、0 warnings、0 hazards | [日志](data/racecheck.log) |

checks 针对有限 scores/V 和 FP32 中间不溢出的定义域。sanitizer 不执行长序列；全员屏障与 union 生命周期仍由所有权/控制流证明支持。

## 1.4 正式 benchmark 与 capture

口径为 GTX1060、FP32、S=16384,d=64、seed0、TF32 off、CPU threads1、5 warmup、20 calls/repeat、5 repeats，无 profiler。均值比均值，中位数单独报告；历史基线与本轮频率/温度/采集轮次仍不同。

| 实现 | non-causal Event 均值 | 中位数 | 来源 |
|---|---:|---:|---|
| V0 归档 | 2922.297 ms | 2916.777 ms | [V0 evidence](../v0/data/evidence.json) |
| V1 归档 | 348.178 ms | 348.262 ms | [V1 JSON](../v1/data/benchmark-custom.json) |
| V2 归档 | 143.560 ms | 143.373 ms | [V2 JSON](../v2/data/benchmark-custom.json) |
| V3 最终 | **47.459 ms** | **47.489 ms** | [V3 JSON](data/benchmark-custom.json) |
| efficient 本轮 | 36.171 ms | 36.180 ms | [efficient JSON](data/benchmark-efficient.json) |

V2/V3=**3.025**，V0/V3=**61.575**，V3/efficient=**1.312**。最终 V3 五组 Event 样本为 47.056、47.486、47.587、47.489、47.678 ms，完整精度保存在 JSON。所有本轮 benchmark 的额外峰值分配为 4 MiB，等于输出，不是总 IO。

[causal](data/benchmark-custom-causal.json) 均值 **25.760 ms**、中位数 **25.713 ms**，相对 V2 causal 快 **2.854 倍**。本轮未做 efficient causal 对照，不能将 non-causal 的 36.171 ms 混入该比较。

无 counters [时间线](data/nvprof-timeline.csv) 捕获单 kernel **47.572188 ms**，grid=(256,1,1)、block=(128,1,1)、128 registers/thread、static shared18 KiB、dynamic shared0。[原始报告](profiles/forward.nvvp) 与身份记录保留；速度比仍用无 profiler 五组计时。

## 1.5 Counters、读写速率与剩余问题

### 1.5.1 发射与等待

同形状 5 warmup 后 capture 一次，nvprof 29 项指标、aggregate-mode on，触发 replay，无 overflow。原始 [CSV](data/nvprof-custom-s16384-d64.csv)、[manifest 的 profile.metrics](manifest.json)、[进程身份](data/nvprof-process.log) 与 efficient 参照均保留。独立 counters JSON 是 CSV 的转换副本，汇总已并入 manifest。

| 指标 | V2 归档 | V3 最终 | efficient 参照 |
|---|---:|---:|---:|
| achieved occupancy | 62.4171% | 23.4428% | 18.6268% |
| eligible warps/cycle | 3.211435 | 4.630517 | 4.324108 |
| issue slot utilization | 50.959040% | 68.346762% | 66.997537% |
| warp execution efficiency | 98.384853% | 100.000000% | 99.573523% |
| exec dependency | 53.507239% | 58.416421% | 37.372316% |
| memory dependency | 6.239920% | 5.328740% | 0.827313% |
| sync | 7.668617% | 3.932961% | 23.071899% |
| instruction fetch | 3.610647% | 9.165196% | 12.554382% |
| other | 24.289828% | 3.497985% | 4.272451% |

V3 的 exec 相对占比仍高，但总时间显著减少；占比不能乘以独立计时变成绝对成本。提高 issue 不要求 occupancy 同时提高，微矩阵提供更多独立乘加结果且复用操作数，是本轮核心变化。

### 1.5.2 DRAM、L2 与 shared 速率

下表每行统一单位，MB/GB 为十进制字节。吞吐直接来自相应 profiler pass，不由逻辑字节/benchmark 推算。

| 操作 | 单位 | V2 归档 | V3 最终 | efficient 参照 |
|---|---|---:|---:|---:|
| DRAM read | GB/s | 2.251748 | 1.233688 | 6.665775 |
| DRAM write | MB/s | 28.126977 | 90.592700 | 109.955744 |
| L2 read | GB/s | 69.605342 | 22.016892 | 82.461766 |
| L2 write | MB/s | 28.258494 | 165.900725 | 109.948200 |
| shared load | GB/s | 888.284720 | 829.218939 | 961.271710 |
| shared store | GB/s | 111.035590 | 62.191420 | 382.001002 |

V3 总时间更短却 L2 read throughput 更低，与结构上减少 global/shared 请求相容；吞吐低本身不是更慢。L2/DRAM 写速率同时受事务量与采集时间影响，不能用比例推导 O 的写入次数。各吞吐来自不同 replay pass，不乘以正式时间来还原“同次字节量”。

### 1.5.3 事务与布局

| 指标 | V2 归档 | V3 最终 | efficient 参照 |
|---|---:|---:|---:|
| DRAM read transactions | 10887482 | 1996852 | 8142739 |
| L2 read transactions | 336550499 | 35636635 | 100733168 |
| L2 texture read hit rate | 94.709465% | 94.373226% | 95.357283% |
| global load efficiency | 100% | 100% | 100% |
| global store efficiency | 100% | 50% | 100% |
| shared load transactions/request | 1 | 1 | 2.796997 |
| shared store transactions/request | 1 | 1 | 3.435517 |

V3 主形状 global K/V 逻辑读取 16→2 GiB，硬件事务降幅不同，受缓存、调度与 replay 影响。V3 shared 每请求事务为1，支持最终布局；efficient 的向量访问可能天然需要多事务，不能将它的较高比值独立解读为错误布局。

V3 的八-lane 输出子组分段写回，gst efficiency50% 是可观察的额外事务；它只发生于最终4MiB输出，仍应先关注主循环矩阵计算与 key tile，而不是按这一比值判断主要瓶颈。

## 1.6 假设核验与下一步

| 假设 | 最终结果 |
|---|---|
| H1：寄存器微矩阵无 spill | 成立，全部模板 local/stack/spill0 |
| H2：global/shared 操作数复用 | 源码结构成立，DRAM/L2事务显著减少 |
| H3：四屏障的 union 生命周期安全 | 源码/SASS一致，三类检查通过 |
| H4：就绪工作与 issue 改善 | eligible4.631、issue68.347%，高于 V2 |
| H5：明确低于 V2 且保持接口 | 3.025倍，全 d1..128、causal/tail验证通过 |
| H6：精度不放宽 | FP64容差不变，主形状20个抽样结果通过 |

V3 与 efficient 剩余约 1.312 倍差距。优先候选为 **减少 P/shared 峰值，让 64-key tile 在全部 head dimension 下成立**，并重新权衡 register fragment、向量化操作数与小范围预取。efficient64-key使每行状态递推减半；run4证明直接放大V3的P/V布局会超过d=128的shared限制，需要重新设计生命周期或按资源分派。instruction-fetch优化仍需和寄存器静态索引一起考虑，不能任意关闭所有展开。

以上只说明记录设备与主形状，不保证所有小d/短S性能同样改善；长序列FP64仅抽样、非有限输入未扩展，历史基线与本轮条件不同。所有消融、失败与最终版本身份固定在 [manifest](manifest.json)。
