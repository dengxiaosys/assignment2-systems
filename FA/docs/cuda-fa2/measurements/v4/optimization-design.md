# 1. V4 优化依据、设计与实验验证

本文件先记录理论推导和验收预测，再实施对应实验。设计时V4尚未达标：run1–run11的最佳主形状结果是run4的46.128 ms，同轮efficient约36.184 ms；这些参数探索保留为反例。理论指导的run12以后实验见§1.5–1.6。最终候选run20为32.924 ms，同轮efficient36.126 ms，完整验收见[性能报告](perf_v4.md)。本文将“源码确认”“模型推导”“实测结果”和“待验证假设”分别表述。

## 1.1 研究范围与来源

### 1.1.1 可以迁移的算法与工程方法

[FlashAttention-2 论文 §3](https://arxiv.org/html/2307.08691v1#S3) 提出减少非矩阵运算、沿 query 维度并行、减少 warp 间中间结果交换。[作者对 sliced-Q 的说明](https://princeton-nlp.github.io/flash-atttention-2/#better-work-partitioning) 明确指出 sliced-K 的 shared 归约开销。论文在 A100/Tensor Core 上的吞吐数字不适用于 GTX1060；可迁移的是在线 softmax、所有权和通信量分析，而不是特定峰值比例。

[CUTLASS Efficient GEMM 的 Pipelining](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/efficient_gemm.html#pipelining) 将低驻留 GEMM 的延迟隐藏分成两个层次：global→shared 的 CTA 双缓冲，以及 shared→register 的 warp fragment 双缓冲。[MmaPipelined 实际源码](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/mma_pipelined.h) 先发出下一组 shared load，再执行当前组乘加，并在当前计算中加载下一阶段 global fragment。[PyTorch v2.6.0 CustomMmaPipelined](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h) 保留相同的两级机制，注释特别强调优先发出延迟要求更紧的 shared load。

[CUTLASS DefaultMmaCore SIMT](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/default_mma_core_simt.h) 为 32×32 warp tile 选择 4×8 线程排列、每线程 8×4 accumulator、最多四元素的 operand 向量访问。[thread MmaGeneric](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/thread/mma_sm50.h) 采用蛇形遍历独立 accumulator，便于操作数复用。这里不能将 SM86 的额外 2×2 zigzag 分支直接搬到 Pascal。

### 1.1.2 Pascal 的限制与可疑指标

[Pascal Tuning Guide 的 Instruction Scheduling、Occupancy 和 Shared Memory](https://docs.nvidia.com/cuda/archive/12.8.0/pascal-tuning-guide/index.html) 确认 GP10x 类 SM 的四 warp scheduler、64K 个 32-bit registers、96 KiB shared/SM、48 KiB shared/CTA，以及固定四字节 bank。设备实际资源由既有测量确认，GTX1060 有十个 SM。该指南建议每 CTA 不超过 32 KiB，以保留三个 CTA 在 96 KiB SM 内的可能性；这不是保证，因为寄存器同样限制驻留。

[Pascal 延迟讨论](https://forums.developer.nvidia.com/t/how-costly-is-the-s2r-instruction-reading-a-special-register/50472/1) 有社区测得的固定指令约六周期，以及 NVIDIA 版主对可变延迟指令的解释。这些数值只作定性参考，不当作本卡实测延迟。[NVIDIA 版主对 Maxwell/Pascal shared 与 texture 路径的说明](https://forums.developer.nvidia.com/t/do-dedicated-shared-memory-and-unified-l1-texture-cache-share-the-same-bandwidth-pascal/47424/6) 说明两者共享 register writeback 资源，global/cache 替代 shared 并不天然更快。

[LDS.128 与 wavefront 的论坛分析](https://forums.developer.nvidia.com/t/clarification-bank-conflicts-metric-vs-wavefronts-for-shared-memory-lds-128/359822/3) 指出四个正常 128-byte 事务不能被等同为四路 bank conflict；[broadcast 特例讨论](https://forums.developer.nvidia.com/t/how-to-calculate-the-bank-conflicts/374947/10) 还显示广播会改变事务基线。上述实验设备包含 Hopper/Blackwell，不能直接断言 Pascal 有相同 broadcast 特例。V4 必须结合本卡 SASS、访问地址和 nvprof 指标，而不能只看 transactions/request 是否大于一。

## 1.2 当前实现为什么没有达到目标

### 1.2.1 工作量和指令比例

忽略边界与 softmax，两个矩阵乘法总计 \(4S^2d\) FLOPs。S=16384,d=64 时为 68.719476736 GFLOPs。这个数字是算法工作量，不是新增测量数据。按 efficient 的约 36 ms 估算有效 attention 吞吐约 1.9 TFLOPs/s，不意味着该值就是硬件峰值或独立 GEMM 实测。

V3 的每线程 QK 微矩阵为 4×4，每个 reduction dimension 需要四次 Q shuffle、四次 K scalar shared load 和十六次 FFMA。PV 是 4×8，每个 key 需要四次 P scalar load、八次 V scalar load和三十二次 FFMA。只计这些主要指令，两个 32-key tile 合计为 5888 次指令/线程；不含地址、循环、mask、softmax 和同步。这解释了为什么扩大 tile 却保留相同通信模式，不能期待成倍加速。

run4 把 key tile 增至 64、P 保留寄存器。对应 QK 是四次 shuffle、八次 K load、三十二次 FFMA；PV 仍需概率广播和 V load。按同样口径约 5632 次指令/线程，主要指令数只比 V3 少约 4.35%。run4 的小幅收益与该模型相容，不能仅用“softmax 轮数减半”预测总时长减半。

### 1.2.2 run11 的具体反例

[run11 源码](run11/snapshot/fa2_forward.cu) 使用 64×64 CTA、四个 32×32 warp 子块，保留完整 K 与大块 Q，没有两级软件流水线。[编译日志](run11/data/build.log) 的主模板为 136 registers/thread、36096 B shared/CTA；[计时](run11/data/benchmark-custom.json) 为 52.387704 ms。

[诊断 counters](run11/data/nvprof-diagnostic.csv) 为 occupancy 12.4104%、eligible warps/cycle 2.524691、issue 49.695397%、memory dependency 29.349599%、exec dependency 43.108759%。shared 容量使每 SM 至多放下两个 CTA，八个 warps；这是理论资源上限，实测 occupancy 与之相容。该证据支持“驻留不足且 load-use 暴露”的假设，但不能把两个 stall 百分比乘以独立计时，分解出绝对成本。

## 1.3 下一轮设计：分阶段搬运与寄存器预取

### 1.3.1 参数由资源和微矩阵推导

保留 run11 的 64×64 CTA、4×8 lane 排列和 8×4 score 微矩阵，先不混入新的 warp 分派。它与实际 efficient 的 CUTLASS SIMT 路径可比。两列 warp 只协作 softmax 的行统计，PV 分别持有不同 output column；不做 output accumulator 的跨 warp 归约。后续是否改为完全 sliced-Q，必须用通信节省和 accumulator/register 增加一起评估。

把 QK 的 reduction stage 定为八个 dimension，来自 efficient 的 SIMT ThreadK=8 和 shared 预算，而不是随机扫描。Q/K 两套双缓冲为 \(2\times8\times(68+68)\times4=8704\) B。PV 保留 P 的 64×64 shared 中间块，V 用两套八-key stage：d=64 时 \(64\times64\times4+2\times8\times64\times4=20480\) B。五组行统计占 1280 B，union 的主模板总量预计 21760 B。精确值以编译的 sizeof 和资源报告为准。

该设计在 shared 方面允许三个 CTA/SM；要使 registers 也允许三个，需考虑八寄存器分配粒度，目标编译值不高于 168 registers/thread，并要求无 local spill。每线程主要常驻状态为 32 个 score 和 32 个 d=64 output accumulator；两套 Q/K operand 共 24 floats。地址、统计与 global prefetch 仍可能使实际寄存器超过估计，必须看 ptxas。

### 1.3.2 两级流水线的依赖关系

CTA 层先发布 stage0，再将 stage1 的 global 数据读入独立寄存器；当前八个 dimension/key 的 FFMA 完成期间可以等待 global 数据。随后写入另一个 shared stage、全员 barrier、切换读 stage。每次复用之前，前一轮所有读者已经通过 barrier。Pascal 没有 cp.async；这里使用普通 load 的异步完成及独立指令调度。

warp 层先加载 reduction index0 的 Q/K 或 P/V fragment，循环中在执行当前外积之前发出下一 index 的 shared load。两套 fragment 使用编译期槽位，不引入动态 register-array 索引。每次外积保留三十二个独立 accumulator，并按 CUTLASS 的蛇形顺序遍历。该顺序影响调度与 operand reuse，不改变每个 accumulator 沿 reduction dimension 的求和顺序。

stage 层循环保留，八次内层循环只展开到双缓冲槽位可以静态确定的程度。不能恢复数千条全展开 FFMA：V3 的历史 SASS/counters 已经证明 instruction-fetch 退化。代码体积、寄存器和发射都列入验收。

### 1.3.3 P 的共享布局由访问公式决定

直接把 P 转置为 P[key][query] 并增加四元素 padding，没有消除 run11 的特殊写入模式：一个 warp 的相邻 lane 各持有四个 key，多个 row subgroup 持有八个 query。scalar store 与向量 store 的冲突必须按其各自服务范围计算。

下一版使用四元素 chunk 保持 float4 对齐的置换：物理 query 下标为 `queryi ^ (((keyi / 4) % 8) * 4)`，P 行宽64。同一 float4 内低两位不变，因此 P/V 消费端仍能一次加载四行。对固定 key，读取的 query chunk 只是置换；对连续八个 column lane 的 store，不同 key chunk 把目标映射到八组不同的四-bank 区间。这是地址模型推导，不是本卡事务数测量；SASS 必须确认实际生成向量 store，必要时做独立访问微基准。

## 1.4 预先声明的验证与否证条件

### 1.4.1 机制预测

第一项预测：主模板 shared 应从36096 B降至约22 KiB，资源应允许三个CTA；如果 registers 超过可驻留档或发生 spill，不能宣称该机制成功。第二项预测：相同工作量下 memory-dependency stall 应降低，eligible/issue 应提高；百分比变化必须结合无 profiler 计时。第三项预测：P 向量 store 生成与地址置换一致，不能用全kernel平均transactions/request单独证明。

### 1.4.2 消融与最终验收

先测完整设计，再分别去掉寄存器 fragment 预取、CTA global 预取和 P 置换，保留相同主 tile 与数值路径。每个消融都保留独立源码、binary、hash、正确性和计时；不要将不同 shared/寄存器驻留变化全部归因于某个源代码行。对照 efficient 用交替顺序的成对重复，主形状须实际追平；seed和温度变化不能只挑有利轮次。

全部 d=1..128、causal/non-causal、tile尾部、softmax最大值跨tile变化和正负抵消保持 rtol=atol=2e-5。长序列参考仍按 CPU FP64 抽样行，明确没有验全矩阵。最终验证还包括多个 S/d、memcheck/synccheck/racecheck，以及实际 extension hash 一致性。

### 1.4.3 复杂度与可并行性

算术复杂度保持 \(O(S^2d)\)，causal 约减少一半有效矩阵区域；临时 shared 为 \(O(B_rB_c+Bd)\)、register accumulator 为 \(O(B_rB_c/T+B_rd/T)\)，输出占 \(O(Sd)\)，不生成 \(S^2\) global 中间矩阵。query CTA 相互独立；每个CTA内部必须顺序进行在线softmax的key tile递推，流水线只重叠数据搬运与独立乘加，不跨越该数值依赖。

沿 key 跨 CTA 的 split-K 只适合 CTA 总量不足等场景，需额外保存并合并 \((m,l,z)\)。当前主形状已有256个CTA、十个SM，缺的是每SM就绪工作与调度效率，尚无证据支持优先增加全局归并kernel。

## 1.5 两级流水线的实测与消融

### 1.5.1 资源与延迟预测得到支持

[run12 编译日志](run12/data/build.log) 确认 d=64 模板为158 registers/thread、21760 B shared、零 spill；d=128 为168 registers/thread、25856 B shared、零 spill。二者均允许三个128-thread CTA驻留。13项测试及长序列抽样FP64验证通过；这只代表已经覆盖的数值案例，不能替代后续sanitizer。

[run12 独立计时](run12/data/benchmark-custom.json) 的CUDA event中位数为38.212402 ms，[同轮 efficient](run12/data/benchmark-efficient.json) 为36.133615 ms，尚慢5.753%。[诊断 counters](run12/data/nvprof-diagnostic.csv) 显示occupancy18.6088%、eligible4.582738、issue70.441669%、memory dependency5.177093%；run11对应值为12.4104%、2.524691、49.695397%、29.349599%。资源与load-use预测均得到支持，但两项机制同时变化，不能据此分配各自绝对贡献。

### 1.5.2 分离机制后的反例

| 实验 | 相对于run12的变化 | Custom event中位数 | 同轮efficient event中位数 |
|---|---|---:|---:|
| [run13](run13/snapshot/fa2_forward.cu) | P取消XOR置换，保持float4写入和相同尺寸 | [40.023807 ms](run13/data/benchmark-custom.json) | [36.193582 ms](run13/data/benchmark-efficient.json) |
| [run14](run14/snapshot/fa2_forward.cu) | global load移到当前stage乘加之后 | [39.215649 ms](run14/data/benchmark-custom.json) | [36.217175 ms](run14/data/benchmark-efficient.json) |
| [run15](run15/snapshot/fa2_forward.cu) | shared fragment改为当前项加载后乘加，保持global预取 | [37.563776 ms](run15/data/benchmark-custom.json) | [36.128717 ms](run15/data/benchmark-efficient.json) |

run13/run14退化，支持保留P置换与global预取。run15反而小幅更快，因此不能宣称源代码显式fragment双缓冲必然有效：编译器会跨展开的外积重排load，run12和run15的寄存器分配、指令体积也不同。后续保留run15结构作为较快基线，同时审查SASS的实际加载提前量。这个结果否证了“必须显式双fragment才能获得warp级预取”的强假设。

### 1.5.3 Base-2 softmax的下一项假设

[PyTorch iterative_softmax](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L1214-L1293) 将scale乘以log2(e)，在最大值和概率递推中使用exp2。当前run12 SASS的每个`__expf`包含log2(e)乘法、RRO、EX2及次正规数处理分支；仅32个score概率即形成较多重复控制指令。下一轮保持matmul、布局和在线递推不变，改用base-2 score与PTX `ex2.approx.f32`，预期减少每个概率的乘法，但不预设一定提速。

[PTX ISA ex2规范](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-ex2) 说明其最大误差为2 ulp，sm20+默认支持次正规结果；`.ftz`会清零次正规输入与结果。因此实验明确不使用`.ftz`，不能通过丢弃微小概率换速度。CUDA12.8没有可用的device `__exp2f`，run16是编译失败记录；后续使用明确的PTX原语，精度阈值保持2e-5。数学恒等式不保证浮点逐位相同，必须重新验证跨tile最大值变化与正负抵消。

## 1.6 向量搬运与最终组合

### 1.6.1 指数实验的实际结果

[run17](run17/data/benchmark-custom.json) 为37.338983 ms，同轮efficient36.142947 ms；base-2转换只带来小幅变化。[run18](run18/data/benchmark-custom.json) 先用warp vote检查指数范围，再在普通范围采用`.ftz`快路径、异常范围保留完整指数，结果38.102527 ms，同轮efficient36.158823 ms。范围检查和重复代码没有净收益，最终删除该分支，保留默认支持次正规结果的PTX原语。

次正规概率不可以无条件清零：例如约`exp(-90)`的概率乘以`1e36`的V仍能产生可见贡献。新增测试在score gap86/88/90/96和跨64-key tile条件下验证这一点。它检验的是attention输出，不是独立指数逐位正确性；最终容差仍为rtol=atol=2e-5。

### 1.6.2 Aligned global access的推导

[CUTLASS SIMT core的访问宽度](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/default_mma_core_simt.h) 和[PyTorch MM0 alignment配置](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L365-L394) 给出向量搬运的依据。这里stage8、64行、128线程，每线程恰好搬运4个float；连续四dimension合成一次float4 load，可减少global指令和地址计算。shared仍转置为dimension×row，避免改变warp微矩阵及在线softmax的数值顺序。

适用条件是head_dim能被4整除，且Q/K/V各自起始地址满足16-byte对齐。连续tensor可以带非零storage offset，所以不能仅检查`is_contiguous()`或head_dim。其他输入使用scalar模板，不限制d=1..128的原契约。d>64时V stage为1024元素，每线程搬运两组float4，Q/K仍为一组。

### 1.6.3 机制证据与最终候选

[run19计时](run19/data/benchmark-custom.json) 为32.908493 ms，[同轮efficient](run19/data/benchmark-efficient.json) 为36.120575 ms。主模板为157 registers/thread、21760 B shared、零spill；寄存器资源档位保持三个CTA/SM。SASS静态指令数由run17的2808降至2448，FFMA数量仍为681，符合“减少搬运和地址指令”的预测。静态计数不等同于动态执行次数，不能单凭该差值推算绝对耗时。

[run20源码](snapshot/fa2_forward.cu) 补充输入起始地址检查，删除消融留下的未使用fragment槽位。[独立计时](data/benchmark-custom.json) 为32.923752 ms，[同轮efficient](data/benchmark-efficient.json) 为36.126306 ms，保持收益。最终组合为64×64 CTA、8×4 score微矩阵、shared生命周期复用、stage8 global预取、编译器调度shared fragment、P chunk置换和对齐向量搬运。性能归因以消融与资源/SASS/counters共同支撑，不能声称任何单一机制独立解释全部提升。

[run21](run21/snapshot/fa2_forward.cu) 在最终源码中仅关闭aligned分派，保持相同softmax和shared路径；[计时](run21/data/benchmark-custom.json) 为37.459711 ms，[同轮efficient](run21/data/benchmark-efficient.json) 为36.267859 ms。run20比值为1.097273、run21为0.968184，经过各自同轮efficient校准，向量版本约快13.33%。这个控制实验补上run17→run19之间softmax代码整理的混杂因素。aligned SASS为6条LDG.128静态位置，scalar为24条LDG静态位置，shared读取与FFMA数量相同；收益仍包含地址计算和编译器调度变化。
