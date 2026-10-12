# 1. V5 优化依据与预先声明的验证

## 1.1 目标、约束与吞吐模型

验收要求为GTX1060、连续二维FP32、S=16384,d=64下，V5耗时不超过同轮PyTorch efficient的60%，causal/non-causal分别比较。按[V4当轮efficient](../v4/data/benchmark-efficient.json)的约36.1 ms作设计参考，non-causal目标约21.7 ms；最后必须重新成对计时，不能把历史时长当成验收对照。该要求相当于至少1.667倍速度，而不是1.4倍。

主形状两个矩阵乘法总工作约为\(4S^2d=68.719477\) GFLOPs，目标有效吞吐约3.17 TFLOPs/s；V4为约2.09 TFLOPs/s。以上是工作量与历史延迟推导，不是V5实测。FP32、d=1..128、rtol=atol=2e-5、不生成S×S global中间矩阵、尾部及不对齐连续输入契约保持。

[V4计数器](../v4/data/nvprof-custom-s16384-d64.csv)显示issue71.99%、memory dependency7.68%、sync17.45%、exec dependency38.51%。DRAM读仅2.41 GB/s，没有支持“优先优化显存带宽就能达到目标”的证据。stall百分比不能乘以延迟来拆分绝对耗时；必须结合指令与独立计时验证。

## 1.2 来源与可迁移机制

[FlashAttention-2作者说明](https://princeton-nlp.github.io/flash-atttention-2/#better-work-partitioning)将sliced-K改为sliced-Q，使warp持有不同query、共享K/V，减少warp间中间结果通信；同时强调减少softmax重缩放、边界检查和因果mask。该文A100 Tensor Core数字不能用作GTX1060性能预测，适用的是所有权和通信分析。

[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/efficient_gemm.html#pipelining)说明CTA共享stage和warp寄存器fragment流水线，以及增大tile和寄存器占用之间的权衡。[SIMT thread MMA源码](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/thread/mma_sm50.h)给出外积与蛇形accumulator遍历，Pascal不使用其sm86专属2×2分支。

[Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/pascal-tuning-guide/index.html)确认每SM64K registers、四warp scheduler、固定4-byte shared bank、GP10x类96 KiB shared/SM和48 KiB/CTA。建议每CTA不超过32 KiB以保留三个CTA的可能性，但实际仍受寄存器限制。FP16在该架构吞吐不优于FP32，不能靠改精度达标。

[SIMT SGEMM工作记录](https://siboehm.com/articles/22/CUDA-MMM)从global合并访存、二维register tiling和warp tiling建立优化路径；其A6000吞吐值不迁移。本版用它交叉确认“每次operand加载服务多个独立FFMA”的指令模型，而不随机扫描大量参数。

## 1.3 第一项设计：warp独立在线softmax

### 1.3.1 所有权推导

保持64×64 CTA和128 threads，把V4的2×2 warp划分改为四个独立query warp。warp各处理16 query×64 key，lane排列仍为4行组×8列组，每线程score由8×4改为4×8。主形状output也为4×8，每线程score和output各32个accumulator，主要寄存器容量没有翻倍。

QK每reduction dimension加载4个Q、8个K，共12个operand float，对应32个FFMA；V4为8Q+4K，同样12个float、32FFMA。因此不能预测仅换布局就使矩阵乘加吞吐翻倍。主要预测是行最大值、分母和alpha由warp局部寄存器维护，不经过shared partial、shared m/l/alpha和三个CTA softmax屏障；每线程维护4行状态，并用8-lane shuffle归约。

P仍写shared供PV读取，但各warp独占自己的query行，K/V仍共享。这是warp独立行统计，不是删除所有CTA通信；global→shared的发布和union复用仍需CTA屏障。

### 1.3.2 第一轮与第二轮消融

第一轮保留stage8，先验证所有权变化。第二轮才增至stage16，使QK/PV阶段屏障由各8次降至各4次。内层只展开4步，避免完整16步展开导致instruction-fetch退化。

stage16下QK双缓冲为\(2×16×(68+68)×4=17408\) B；PV为\(64×64×4+2×16×64×4=24576\) B，union主模板24 KiB，未使用shared行统计。d=128的PV为32 KiB。主形状shared仍允许三个CTA/SM，但若编译寄存器超过可驻留档、发生spill或fetch退化，必须调整模型，不能把预期当作结果。

P chunk置换延续V4四元素对齐规则，warp每行组4个query，每lane拥有两组4个key。consumer分两组输出列读取，每个概率只在自身query warp内消费；读写布局需以SASS和事务地址核对。

## 1.4 后续候选与否证条件

如果独立softmax有效而矩阵吞吐不足，下一项从两类机制选择：保留Q跨key tile以减少重复搬运，或增大warp微矩阵以提高FFMA/operand指令比。前者增加shared/register容量，后者增加accumulator和spill风险；实施前都要计算资源上限。V4已有256个CTA，split-K增加全局归并当前不是首选。

所有实验先写假设、再编译资源与数值测试、最后独立成对计时。至少对所有权、stage、布局或持久数据复用做可解释消融。失败版本保留源码、状态和原始数据；未达到60%目标不得结束为成功。

最终验收包含三个seed交替顺序成对计时、多个S/d、完整小形状CPU FP64与长序列抽样行、memcheck/synccheck/racecheck、29项counters、SASS/resource与实际扩展哈希。旧V3/V4直接引用既有归档，不覆盖原始证据；不自动提交Git。

## 1.5 复杂度与可并行性

算术复杂度保持\(O(S^2d)\)，global新增输出为\(O(Sd)\)，shared为\(O(B_rB_c+Bd)\)，register accumulator为\(O(B_rB_c/T+B_rd/T)\)。query CTA互相独立；warp分别拥有query行，在线softmax仍按key tile顺序递推。流水线重叠搬运与独立外积，不改变最大值/分母/输出的数值依赖，也不跨query做近似。

## 1.6 前两轮结果与8×8微矩阵假设

run1的sliced-Q stage8为32.314352 ms，run2 stage16为32.208487 ms，对照run2 efficient36.160248 ms。二者均通过15项数值测试。主模板寄存器约155→154、shared20→24 KiB、零spill，stage变大没有足够收益，说明不能把V4同步stall当成可独立消除的绝对成本。

下一轮把d≤64主模板query CTA扩成128，每warp32query，线程score/output微矩阵由4×8变8×8。每dimension的主要操作数由12个float、32FFMA变成16个float、64FFMA，operand/FFMA比由0.375降至0.25；shared向量load主要指令由3/32降至4/64。输出accumulator64、score64、行状态16，预计寄存器接近200–240，但必须编译确认。

P为128×64、V双stage16×64，总shared40 KiB，至多两个CTA/SM。低驻留的代价需由64个独立accumulator与更少load来抵消，依据CUTLASS对register tiling和occupancy权衡的说明，不能预设越大越快。d>64保留64query与stage8，防止output128+score64的寄存器溢出。保持相同因果/softmax算法，若spill或独立计时退化则否证该配置。

run3实测29.868536 ms，对照36.170215 ms，15项测试通过且零spill；主模板254–255 registers/thread、40 KiB shared，至多两个CTA，超过事前200–240的估计。矩阵复用有效但尚不足目标。

下一项把同样64个score accumulator用于64query×128key CTA：lane改为2行组×16列组，每线程QK仍8×8，PV变8×4，output accumulator由64降32。softmax轮数减半，score仍64，但PV operand/FFMA比从16/64回到12/32，因此不能假定净收益；预期寄存器下降允许更好的调度。shared P仍32 KiB，V双stage16×64为8 KiB，总40 KiB。该矩形tile选择来自score/output资源平衡，而非随机扫描。

run4矩形tile为31.384882 ms，对照36.145459 ms，主模板226registers、40 KiB shared、零spill。虽然寄存器下降，PV复用退化后的总时长仍更差。

[Scott Gray的Maxwell SGEMM解析](https://github.com/NervanaSystems/maxas/wiki/SGEMM)以64-thread、每线程8×8外积、双stage8软件流水线为小tile方案，并详细讨论operand reuse和向量shared访问的实际限制；其GM204吞吐不可直接当成本卡数据。下一轮保留run3的每warp32×64、线程8×8，但只用两个warp组成64×64 CTA，stage回到8。每CTA总score/P减少一半，主shared约20 KiB，预计寄存器约220–250，允许四CTA/SM、八warp；active warp数量未比run3增加，预期收益来自CTA调度粒度、P容量和更小stage开销。若性能没有净收益，后续重点应是shared地址映射和机器指令调度，不继续只放大tile。

run5为34.532300 ms，对照36.209048 ms，主模板237registers、20 KiB shared、零spill，15项测试通过。[诊断CSV](run5/data/nvprof-diagnostic.csv)显示fetch stall21.100479%、memory dependency3.169677%、issue69.648760%。动态warp instructions约1.5643e9，FP32 FMA约3.4399e10个thread级操作；FMA换算warp约1.0750e9，其余指令占明显比例，两个统计的predication口径不同，不能直接按差值分摊时长。

下一项依据FA2减少边界与非矩阵运算的原则，将逐score合法性检查只放在query/key尾部或causal对角tile的CTA一致分支内；完整interior tile不逐元素检查。最终每输出scalar的除法改成每行一次`1/l`并乘各输出。后者改变最终舍入路径，必须保持2e-5校验，包括大V和抵消案例；不做fast_math或概率裁剪。预计控制和重复除法静态体积下降，是否净提速以run6验证。

run6为31.759564 ms，保持数值通过，主要资源档位未变。下一轮只把lane地址从连续8-lane行组改成交错排列：`row_lanei=lanei%4`、`col_lanei=lanei/4`，归约shuffle距离相应乘4。4个query组和8个column组的逻辑所有权不变，但每个LDS.128的8-lane服务集合变成交错A/B quad，这与Maxas的shared广播zigzag原则相关。当前本卡shared tx/request=3并不自动证明三路冲突；地址调整仅作为可否证假设，需重新计时和采集同口径counter。

run7交错lane为32.997989 ms，对照36.128461 ms，数值通过，未获得净收益。当前最佳仍run3约29.87 ms，目标尚未达到。下一阶段依据FA2减少边界/非矩阵指令的原则，研究精确head dimension及完整tile的编译期路径：主循环运行时head_dim产生重复乘法、dimension范围判断与部分tile谓词，aligned仅消除了scalar load数量，没有消除这些控制。精确维度、causal开关和完整序列分块条件可以在host选定，不改变输入或数值算法；必须保留任意d与尾部通用路径，并用编译资源、实际SASS和对照实验判断净收益。

## 1.7 编译期完整分块路径

run8恢复run3的128×64 CTA、4 warp、每线程8×8、stage16，再合入run6的interior mask与每行inverse denominator。host仅在d=64、S%128=0、Q/K/V地址16-byte对齐时选择精确维度与causal模板；其余输入使用运行时维度通用路径。完整query/key tile允许编译删除load的范围判断，行跨度64允许用移位代替运行时整数乘法，non-causal允许编译删除整个score mask区域。

完整tile且每个query至少有一个合法key，使首块m_new有限；初始m=-∞可直接ex2(-∞)=0，后续m保持有限，因此该路径无需每行isfinite分支。对causal的对角tile仍保留逐元素mask，未改变有效attention区域。此推理假设有限Q/K及可表示的dot product，与现有数值测试输入一致；不采用fast_math、概率截断或降精度。

共享容量仍为40 KiB，预期资源收益来自控制流和address temporaries减少；未预设跨过register驻留档位。作为组合实验，若有效还需要拆分消融以区分run6的控制改动和精确维度路径。验收仍为数值、零spill、实际延迟和SASS，而非仅看C++减少的判断数量。

run8为32.100250 ms，对照36.057617 ms，数值通过；专用路径出现32 B stack、64–96 B spill stores、96 B spill loads，255 registers。静态non-causal函数少了大量mask/除法控制，但这没有抵消spill与调度变化。

run9仅把QK/PV的dimension/key内循环从展开4步改为单步循环。run8 SASS显示编译器将多个dimension的operand与accumulator rename保持存活，8×8 accumulator两组已经占128个float，再叠加fragment和地址状态很容易触及255寄存器硬上限。单步循环的预期收益是缩短live range、降低fetch体积和消除spill，代价是更多循环/地址指令；不预设一定提速。这是依赖run8实际编译结果作出的寄存器压力实验。

run9为37.324670 ms，对照36.093265 ms；通用路径降至247–249register且零spill，专用路径仍255register并spill。单步循环未改善主路径。

run10恢复4-step unroll，用一个inline PTX依赖图表达64个`fma.rn.f32`。每个2×2 accumulator按swirl顺序使用两个A和两个B，整个8×8采用蛇形pair遍历；依据Maxas关于两slot operand reuse和编译器重新调度的讨论，预期减少跨外积的accumulator rename与增强reuse，但PTX不固定物理寄存器、也不保证SASS顺序。每个accumulator仍按原reduction顺序更新，明确round-to-nearest FMA与原默认融合乘加一致；不改变精度或数学工作量。需要检查SASS是否真正减少rename/spill，并独立计时。

run10为32.130368 ms，对照36.077451 ms，数值通过，专用路径仍spill，PTX依赖图没有控制ptxas的最终rename。

## 1.8 Maxwell式两warp矩阵映射

run11进一步采用Maxas的完整64-thread 64×64映射：query quad为`((tid>>1)&7)*4`，column quad为`(((tid&0x30)>>3)|(tid&1))*4`，每线程第二quad都相隔32元素。每warp覆盖32种query和32种column的笛卡尔积；两个warp共同覆盖所有64column，因而softmax需跨warp共享maximum及denominator。8×8 score和output accumulator保持，主shared约22 KiB，global fragment各8float。

该映射使Q和K每个shared vector在warp内使用四组quad广播集合，区别于run5的query连续8元素、key相隔32元素的非对称布局。shared probability保留transposed 68-stride形式供PV以相同矩阵映射读取，输出为float4。资源预测为200–250register、四CTA/SM，增加两次softmax CTA同步是明确代价；若矩阵调度收益不足则否证，不能把Maxwell SGEMM的98%理论吞吐套用到包含softmax的attention。此路径仅用于完整d64，尾部及其他d沿用通用kernel。

run11为28.347366 ms，对照36.020257 ms，15项数值通过，247/249register、22 KiB shared、零spill，结果优于run3。SASS仍在单个dimension开始前聚集LDS，随后大量FFMA，说明可以检查register prefetch。

run12引入双register fragment，提前加载d+1后更新d的8×8 accumulator，内循环展开2步以便编译器消除even/odd array索引，同时使主体约128FFMA而非256。额外operand16float可能再次使寄存器超限，预期需由更短展开live range抵消；若发生spill或时长退化则不能采用。该试验针对Maxas与CUTLASS都强调的shared→register流水线，保留相同global/shared stage和softmax算法。

run12无效：build log没有nvcc步骤，实际extension哈希与run11相同，setuptools未跟踪新增`.cuh`依赖；28.674728 ms只属于run11二进制再次计时，不能归因于双register fragment。build entry补充头文件depends，实验runner使用`--force`；run13重新编译相同run12假设，保留run12证据并标注无效，禁止覆盖或混入有效消融。

run13有效重编译为29.675214 ms，对照36.019662 ms；251/252register、零spill，仍退化，恢复run11单fragment/4-step展开。

run11诊断显示shared load tx/request1.969925、memory dependency1.659576%、fetch13.761639%、exec dependency34.425849%、动态指令1.412608e9，最佳映射收益包含更少事务与指令，不支持优先继续增加访存流水线。run14将Q/K/P的shared stride从68改为64。该映射每次LDS读取固定dimension的quad广播，去掉padding不改变单次warp地址bank集合；允许stride乘法降成移位，降低计算地址成本与shared容量。转置global→shared scalar store可能存在原有冲突，故必须实测不能只依据LDS推断。stage8、矩阵映射与归约不变。

run14为28.944267 ms，对照36.038126 ms，数值与零spill通过，没有净收益，恢复stride68。

## 1.9 推迟分母归约

run15只在最终输出前合并denominator。每线程key子集的分母为`l_t=alpha_t*l_(t-1)+sum(exp(score_t-m_t))`，同一query的所有线程共享m_t/alpha_t，线性分配律保证最终各子集l之和等于每tile归约后的递推结果（实数算术）；FP32求和顺序不同，必须检查既有2e-5精度、subnormal×大V及跨tile最大值变化。每tile省两次shuffle、shared partial写与两次读、合并加法；最终仅一次归约与CTA屏障。global矩阵工作量、P/V计算与最大值同步不变，寄存器数量预计不变。

run15为27.896804 ms，对照36.115195 ms，235/242register、零spill、15项数值通过。

## 1.10 持久Q与完整K的共享容量模型

run16使用64query、4warp sliced-Q，每线程4×8 score/output，使accumulator总量从128降64。Q在CTA开始一次性转置到shared并跨所有key tile保留；每tile完整64×64 K转置到shared，QK的64dimension计算只需发布K和结束QK两个CTA屏障，原stage8的8个reduction屏障消失。

Q为64×68×4=17408 B，union中K为17408 B或P17408+V双stage8×64×4=4096 B，合计38912 B，允许两个CTA/SM、八active warp。QK operand/FFMA从16/64退化为12/32，寄存器预计140–180；减少Q重复搬运与QK阶段同步能否抵消微矩阵复用退化必须实测。P布局采用warp独占query的四元素XOR置换，分母继承推迟归约，每warp独立m/l，无跨warpsoftmax同步。仅完整d64走此路径；其余输入通用实现不变。

run16为38.059729 ms，对照36.137450 ms，114/120register、38 KiB shared、零spill，数值通过但退化；低寄存器数量不能补偿微矩阵复用下降。

## 1.11 保留subnormal结果的指数范围变换

run17恢复run15，针对SASS中每个`ex2.approx.f32`产生的`FSETP(x≥-126)`、条件半缩放、RRO、MUFU和条件平方，使用`2^x=2^(x+64)*2^-64`。online softmax传入x≤0，FP32可表示非零2^x所需x≥约-150；平移后[-86,64]的指数结果均为normal，因此可以在SFU上用`ex2.approx.ftz`，最后`mul.rn.f32`不带ftz并产生原本的subnormal结果。x<-190时平移SFU可能flush，但最终真实结果已远低于最小FP32subnormal，正确舍入应为0。

有限x在x+64操作中的绝对舍入误差上界约3.82e-6，其引入相对指数误差约2.65e-6；输入x原有误差、SFU近似和最终乘法舍入仍需数值实测。此改写没有裁剪概率，没有把subnormal概率直接设0，也没有改V/accumulator精度；保留既有极小概率×大V、抵消、长序列FP64校验。若2e-5不通过则不能采用，即使性能改善。只用于完整d64专用路径，通用路径保留原指数。

run17为27.433392 ms，对照35.968983 ms，235/242register、零spill，现有15项校验通过。SASS的72个FSETP指数保护消失，但RRO仍存在。

## 1.12 显式output存储以复用accumulator寄存器

run18在QK阶段让score和上tile output不同时保持64float存活：output暂存最终O张量，QK/softmax把P写shared后读回O再重缩放与PV；线程输出区域互不重叠，最后一次归一化前再读回。global store/load使用volatile PTX使编译器不能把临时O保留为register，不分配S×S矩阵。主形状额外约2 GiB aggregate read+write，O工作集4 MiB大于本卡L2，可能造成带宽退化；这是明确代价，不能先验宣称驻留提升必然更快。

预期QK/PV两组accumulator各自只有64float活跃，寄存器约160–190；shared仍22 KiB，可能允许四CTA保持或更低register但shared限制仍四CTA/SM，active warp上限不增加。主要预期收益是减少rename和register bank压力；若仍寄存器高或L2搬运退化则否证。O存取均FP32，不改变逐keytile的数学更新顺序。

run18为36.062240 ms，对照36.084953 ms；168register、零spill，15项测试通过，额外global访问退化，不能采用。当前最佳run17约27.43 ms，目标未达到。

## 1.13 一次性Q/K分块转置

下一轮恢复run17后，用单独kernel对Q/K做O(Sd)的64-row tile转置，得到`[S/64][64dimension][64row]`连续layout。该额外global scratch为2Sd个FP32，不生成S×Sattention矩阵，端到端计时必须包含准备kernel与scratch分配。每stage的Q/K搬运从row-major vector load加scalar shared转置，变为连续vector load加vector shared store；stage内全CTA仍读取完全相同Q/K。所有权、8×8矩阵块、P/V与softmax不变，目标是减少每tile重复的scalar STS、冲突和复杂地址计算。

该思路来自GEMM的数据打包/转置以换取主体向量访存；需补充官方矩阵转置来源与实际SASS证据。准备是O(Sd)，主体是O(S²d)，主形状每Q/K有约百万元素，准备成本应以实测判断。短序列可能得不偿失，因此专用dispatch仍需多形状验收。Q/K transpose只重排FP32bit，不改变dot product的reduction顺序。

run19的准备kernel采用[NVIDIA官方矩阵转置示例](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)的32×8 thread block与32×33 shared tile，保证输入和输出都连续访问，shared列读取有一个元素padding。每64×64块分成四个32×32块，Q/K在grid.z中区分；主体每stage的Q/K shared写由16条scalar降为4条vector，保留68stride与现有屏障。主形状scratch为8 MiB，端到端benchmark通过同一forward入口分配scratch并启动准备与主体，不在计时外缓存打包结果。编译寄存器约219且零spill；性能仍待独立计时。

最终数值测试扩展完整d64、S=128/256/512路径，避免仅用S=65的次正规概率与最大值变化案例误认为专用kernel已经覆盖。次正规概率乘大V的测试同时覆盖同tile及跨tile重缩放；常规随机输入与成对抵消做全输出CPU FP64比较。

run19端到端27.105432 ms，对照36.090778 ms，约75.1%耗时。诊断中主体动态指令1.3566e9、memory dependency0.601088%、fetch14.944555%、exec dependency30.611078%、issue74.893922%；shared vector写的transactions/request3.665510包含128-bit warp服务，不直接等于冲突倍数。打包减少指令而非显著提升整体吞吐，下一轮不继续扩大访存流水线。

run20仅把8步外积循环从展开4步缩到2步，保持单fragment。run13同时使用双register fragment与展开2步未获收益，不能独立否证缩短展开；当前主函数静态FFMA537条，编译仍把多步accumulator分配为不同物理寄存器。预期降低fetch与live range，代价为每128FFMA增加循环控制；必须独立检验。

run20为28.622821 ms，对照36.099380 ms，216/219register、零spill、16项数值通过；缩短展开退化，恢复4步。run21将专用Maxwell布局的global/shared stage增为16，8个QK和8个PV屏障降至各4个。shared从22528升到26624 B，96 KiB/SM仍能容纳三个CTA（run19四个），active warp从8降6；global fragment各从8升16float，可能提高寄存器。run2的sliced-Q stage16无收益不能直接代表本映射，run19的sync14.94%提供进一步验证理由，但驻留减少是明确代价，失败则不保留。

run21为30.422110 ms，247/248register、26 KiB shared、零spill、16项数值通过；stage16退化，恢复stage8。

run22仍为64-thread、64×64、每线程8×8，但每warp独占32query，使用两个相隔16元素的query quad、两个相隔32元素的column quad。lane的query组为`(lane>>2)&3`、column组为`(lane>>4)*4+(lane&3)`，shuffle距离1/2/16，每warp完整覆盖所有key/output columns。区别于run5的连续8query，本映射仍以四元素quad读取，保持对shared vector广播模式的控制。取消shared最大值/分母，softmax不跨warp且每tile少一次CTA屏障；主shared21 KiB仍四CTA/SM。QK/PV阶段继续CTA协作搬运，P也仍在CTA屏障后消费，不能假定整个CTA无需同步。依据FA2 sliced-Q的通信模型，减通信的净收益需用实际bank事务、寄存器和延迟判断。

run22为28.355371 ms，218/219register、21 KiB shared、零spill、16项数值通过；取消跨warp通信未抵消映射变化，恢复run19映射。run23使用CUDA12.8 ptxas官方`--register-usage-level=0`，其[编译器文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html#ptxas-options)与本机`ptxas --help`说明低级别抑制增加寄存器的激进优化。run19 SASS仍把一组64score在展开reduction中rename至约R180–R217，因此测试降低激进优化能否减少rename和register bank相关调度；不固定寄存器数，不把低寄存器本身当目标。此选项为版本相关beta特性，命令必须归档；数值、spill与计时仍分别验收。

run23为27.174081 ms，对照36.106680 ms，215/219register、零spill、16项数值通过，未获明确收益。

## 1.14 Pascal机器指令的yield分配

run24恢复默认ptxas，先只调整FFMA的warp yield hint，不改任何指令、寄存器、stall或dependency barrier。[MaxAs control codes](https://github.com/NervanaSystems/maxas/wiki/Control-Codes)解释Maxwell每三条指令对应一个64-bit控制字、每slot的bit4是yield，且在64FFMA块中间稀疏yield可以防止一个warp过早抵达屏障。[CuAssembler](https://github.com/cloudcores/CuAssembler)明确支持SM61；其`CuControlCode.py`独立复用相同bit4与21-bit分槽方式，支持将该编码用于本卡。yield不改变程序可见的指令依赖，性能必须由实测判断。

run19主体537条静态FFMA中约98条带yield，多数集中在无数据依赖的外积段。本实验将所有FFMA设置为不yield，仅每32条保留一个yield。脚本校验ELF64/sm61、实际disassembly与机器码一致，保留其他控制位与instruction words，定位扩展内未压缩cubin后等长替换，写入前后SHA256与修改位数。不能把机器指令patch隐藏在源码哈希之外，最终若采用需在标准build入口明确加入并归档脚本。先做数值验证再计时。

run24为30.498093 ms，对照36.043649 ms，16项数值通过；稀疏yield退化，下一轮恢复ptxas的yield策略。

run25仅重新计算连续register-FFMA段内的reuse flags。依据[MaxAs operand cache实现](https://github.com/NervanaSystems/maxas/blob/master/lib/MaxAs/MaxAs.pm)，每个source slot缓存两组操作数，出现后续同slot读取时给前一指令置reuse；写同一register或遇到任意非FFMA立即清空候选，保守避免跨memory/control-flow生命周期。保留寄存器、FFMA順序、yield、stall、屏障及ELF信息，改变的只有slot17–19复用位。SASS静态分析在528条register FFMA中约有87次额外bank读取、398次cache命中；该分析不含硬件operand collectors和memory延迟冲突，不能当作真实stall时间。此消融检验ptxas原reuse是否遗漏同slot双项复用机会，性能与数值分别验证。

run25为27.103412 ms，对照36.048999 ms，16项数值通过，与run19近似，没有采用机器码改写的证据；恢复标准编译。

## 1.15 保持8×8微矩阵的持久Q

run26与run16不同，保留Maxwell 64-thread、每线程8×8 score/output、跨warp最大值和推迟分母。一次性将已打包Q的64×64 tile搬到shared，每key tile完整打包K进入shared，reduction dimension内无CTA屏障。Q为17 KiB，union为P17 KiB+V4 KiB（或K17 KiB），加统计1 KiB，共39 KiB；shared限制两个CTA/SM、四warp，寄存器预测约200–230但不决定驻留。先K发布屏障，归约m后、覆写K之前再设屏障，确保跨warp读取全部完成；P/V的发布与阶段屏障不变。总QK屏障从8次下降到两次，与重复Q搬运的减少一起必须补偿驻留减半。此实验隔离run16中4×8微矩阵复用退化，失败则表明这张卡当前低驻留下更需要阶段调度。

run26为47.434018 ms，对照36.170523 ms，194/199register、39 KiB shared、零spill、16项数值通过。减少屏障/搬运不能补偿驻留减半，恢复阶段加载。

## 1.16 单warp CTA消除跨warp阶段等待

run27由一个warp负责32query×64key，每线程仍8×8 score/output，query两个quad相隔16、column两个quad相隔32。Q双stage为2×8×36，K为2×8×68，P为64×36，V双stage为2×8×64，union最大13 KiB，寄存器预计约220–250、shared允许七CTA/SM（七warp）。warp完整覆盖矩阵，softmax仅需shuffle1/2/16，QK/PV搬运由同一warp完成，因此使用`__syncwarp`发布shared，Pascal编译不需要CTA级屏障；必须synccheck/racecheck验证warp同步与shared生命周期。

相比run22两warp独占query但协作加载，本方案消除stage间跨warp等待，代价是每warp重复完整K/V加载、global fragment从8增16以及CTA数量翻倍。本卡run19 memory dependency0.6%，访存冗余有余量，但不能认为必然有收益。packed Q保留64row布局，32query CTA选择对应半块，所有搬运仍float4。causal对角64key块可能覆盖32query半块，mask以绝对query/key比较，严禁误用原64query对角条件。

run27为34.292685 ms，对照36.098453 ms，242register、13 KiB shared、零spill、无CTA barrier、16项数值通过；消除跨warp阶段屏障仍退化，未采用。

## 1.17 QK/softmax与PV的warp流水线

run28依据[FlashAttention-3 §3.1–3.2](https://arxiv.org/html/2407.08608v1#S3)和[CUTLASS warp specialization](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization)分离producer/consumer。但本卡没有TMA、WGMMA或setmaxnreg，不能使用论文的Hopper吞吐或动态寄存器优势；这里两组各64thread分别执行QK/softmax和PV，P双缓冲连接同一query CTA，producer tile t和consumer tile t−1同轮执行。每线程8×8 accumulator只属于一类矩阵阶段，不再同时持有score和output。两组静态分支分别编译完整循环，防止单warp执行两种角色。

shared为Q/K双stage8共8704 B，P双buffer32768 B，V双stage4096 B，m/l/alpha1536 B，总47104 B=46 KiB，小于48 KiB/CTA；96 KiB/SM允许两CTA、八warp。预计寄存器约140–180（需编译），比run19四CTA×两warp并未增加warp总数，主要收益候选为更短live range与软max/SFU、shared搬运和另一组FFMA重叠。所有warp共享SM的FP32管线，不能把QK/PV并行等同于矩阵工作量减半。

组内用`bar.sync 1,64`和`bar.sync 2,64`发布stage，跨组每tile末`bar.sync 0,128`发布P与alpha并确认前一buffer已消费；双buffer保证producer写t时consumer只读t−1，m/l仅producer使用，最终l在最后发布后交consumer归一化。不使用自旋flag，不依赖未同步shared轮询，不生成S×S global，仍需sanitizer实际验证。

run28为30.126797 ms，对照36.109082 ms，约164register、46 KiB shared、零spill、16项数值通过，未优于run19。run29只限制ptxas `--maxrregcount=128`：每角色实际accumulator64、producer行状态16、operand/global fragment约32、少量地址状态，128附近可能通过更少rename实现；若spill则该预算假设失败。shared仍两CTA/SM，寄存器下降不会提高驻留，所求是机器指令调度而非occupancy。此资源边界实验不改变算法或重新调参多项；必须记录spill与计时，不能因为限定值更整齐而采用。

run29为41.955515 ms，对照36.240436 ms，主模板112 B stack、112–116 B spill loads、220–224 B spill stores，128register预算失败，16项数值通过但不采用。当前最佳仍run19约27.11 ms，60%目标尚未达到。

## 1.18 独立FFMA段的调度与寄存器bank模型

下一轮恢复run19源码与默认ptxas，再只对连续、彼此独立的register-FFMA段调整顺序。每段要求不同destination，且某指令写入的register不作为其他指令的A/B/C输入；遇到memory、branch、waitmask或非一周期stall时切段，禁止移动任何依赖屏障和变量延迟指令。相同output元素的dimension累加顺序不变，段内仅重排不同output的独立乘加，FFMA仍round-to-nearest。branch指向的地址也必须切段，metadata与段外instruction words保持一致。

优化目标依[MaxAs register bank与operand reuse解析](https://github.com/NervanaSystems/maxas/wiki/SGEMM#calculating-c-register-banks-and-reuse)建立四bank、每source slot双项缓存的调度模型。该细粒度bank规则为社区来源，暂标单一来源模型，不把预测extra reads当成真实stall。候选顺序用确定性的局部搜索减少同bank未缓存source读取并保留双项复用；所有control的stall/wait/read/write barrier保持原地址，reuse重新按新顺序计算。若合法性证据不足、数值失败或时长无净收益则回退，不能以模型分数代替测量。

run30额外保留每独立段末尾六条指令，避免推迟的FFMA结果越过段外使用的固定延迟边界。两模板分别移动190/213条instruction word，局部模型的bank额外读取从24/29降到4/5；16项数值通过，27.056772 ms对照36.049402 ms，与run19差异很小，没有采用机器码改写的充分证据。

## 1.19 单概率缓冲与更高warp驻留

run31恢复run28的两角色流水线，但将双P缓冲改为单P缓冲。producer先在register完成QK、maximum和softmax，此时consumer仍读取前一tile的shared P；新增全CTA屏障等待consumer完成，随后producer写下一tile的P和alpha，再用原发布屏障交consumer。两角色在循环中每轮都执行同样两次全CTA屏障，启动和排空轮也保持计数一致；前一alpha在consumer完成前不能被覆盖，producer用8float暂存新alpha。

shared由47104 B降至30464 B（29.75 KiB），96 KiB/SM允许三个CTA，即12warp。寄存器若超过约168/线程则64K寄存器限制可能将驻留退回两个CTA，因此必须看ptxas资源。预期收益是更高驻留与QK/SFU和PV重叠，代价是增加一次tile屏障及alpha的register live range。该实验仍沿用FA3/CUTLASS流水线原理，但针对Pascal无动态寄存器分配调整shared资源；不借用Hopper性能数字。单buffer协议必须精度及sanitizer实际验证。

run31数值16项通过、151/161register、零spill，25.796237 ms对照36.079974 ms。但synccheck报告跨角色在不同PC执行全CTA `bar.sync`的divergent threads，因此该版本不能通过最终验收。诊断occupancy18.46%、issue77.98%、exec19.45%、fetch15.75%、sync24.38%，仅作为假设线索。run32将循环改成所有线程共同控制的统一循环，全CTA屏障使用共同位置的`__syncthreads`；组内stage发布仍采用64thread的named barrier。producer的score与consumer的output使用同一64float变量但分别属于不同线程，producer把新P与alpha保留到共同消费完成屏障后才发布。修复必须重新数值、synccheck及计时。

run32的synccheck为0 errors，16项数值通过，150/152register、零spill，但时长27.707489 ms对照36.089703 ms。run33保留独立角色循环，跨组消费确认和发布分别使用named barrier3与4，显式count128，避免复用特殊的默认barrier0；必须重新synccheck，若仍报错则不能采用分离循环。组内仍ID1/2且count64。该轮仅改变屏障ID/控制布局，不修改软max算术，目标是检验Pascal合法named-barrier producer/consumer协议。

run33为26.456677 ms，但synccheck仍报告256 errors，分离循环未通过验收，恢复run19的普通CTA控制。

## 1.20 FA4条件重缩放的FP32迁移

run34依据[FA4论文§3.1.4](https://arxiv.org/html/2603.05451v1#S3.SS1.SSS4)及[作者SoftmaxSm100实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py)采用阈值8（base2）。维护归一化基准b：新tile最大值≤b+8时继续使用b，P=2^(score−b)，直接累加分母和PV；超过阈值时b改成新maximum并按2^(b_old−b_new)缩放已有状态。这是使用不同softmax平移基准，实数算术下分子/分母同时乘同一因子，因此无需裁剪概率或改变attention工作量。第一tile b=-∞必然触发重缩放。b每次仅在阈值越界时变化，不能用相邻tile差值替代相对于b的差值。

P最大为256，指数范围变换x+64的上界变成72，仍为normal；最小可表示概率继续由非ftz乘法产生subnormal。denominator≤256S，32-bit序列范围内不会溢出，但任意极大V的未归一化分子可能比传统max基准大256倍，这是明确数值风险；必须以大V/抵消/跨tile最大值更新和2e-5测试否证，不可借用FA4 BF16误差门槛。仅d64完整tile走此变体，其余仍通用FP32。条件scale先独立测试，后续再考虑CTA一致跳过maximum通信，不同时加入多个改动。

run34为26.978601 ms对照36.112332 ms，218/219register、零spill、16项数值通过；只做条件重缩放没有明显改善。

## 1.21 合并指数的仿射变换

run35保留run34的条件基准，score在QK结束后保持未缩放。因scale>0，max(score×scale)=max(score)×scale，先每行归约未缩放score再仅缩放行maximum。计算bias=64−b后，每score用`fma.rn.f32(score,scale,bias)`直接得到指数平移输入，替代逐score的FMUL、FADD减基准和FADD加64。MUFU前的RRO不能省略：[NVIDIA论坛的架构解释](https://forums.developer.nvidia.com/t/sin2pi-intrinisc/348014/2)说明sm70之前RRO.EX2把浮点转换成MUFU需要的fixed-point格式，此为社区解释并与本机SASS配对一致，不能把Volta省RRO迁移到Pascal。

FA4作者`scale_subtract_rowmax`同样以FMA组合scale与bias，但其packed FP32指令本卡不可用。此轮只有普通FP32 FMA且指数保留非ftz输出乘法；融合改变舍入，bias=64−b也引入行一致的舍入误差，需2e-5和subnormal/抵消验证。预期每64个score少约128条scalar FP32指令，两个矩阵乘法的work与累加顺序保持；不能按静态指令减少比例声称时长同比下降。

run35为26.773248 ms对照36.127332 ms，219register附近、零spill、16项数值通过，融合小幅改善但不足目标。

## 1.22 CTA统一判断后跳过maximum通信

run36先每线程计算其8行的局部maximum，判断是否有score超过该行基准b+8。`__syncthreads_or`合并所有线程的判断，返回CTA一致结果。如果没有越界，所有行基准均可保持，省两次row shuffle、shared maximum发布/读取及跨warp合并；如果有越界才执行原maximum归约和一次发布屏障。每tile仍计算全部P与PV，无概率跳过或近似。

该模型把64个不同query的条件合成一个CTA分支，可能因任意一行越界而让全CTA走慢路径；随机长序列首块后常少于8的增长只是待测假设，不能预设真实输入分布。每tile增加一个CTA reduction barrier、8float保存局部maximum与分支控制，可能抵消通信减少；需要计时与资源否证。首次tile必走慢路径，并保留causal mask、subnormal测试。

run36为26.797363 ms对照36.106342 ms，218/220register、零spill、16项数值通过；CTA条件maximum通信未取得净收益。

## 1.23 更短stage跨过四CTA驻留边界

run37从通过synccheck的统一角色循环run32出发，将producer Q/K及consumer V的stage缩成4，外积只展开2步。P单buffer仍16 KiB，Q/K/V双stage总容量由12800 B降至6400 B，加统计1280 B，共24064 B=23.5 KiB，96 KiB/SM可容纳四CTA、16warp。register必须≤128/线程才能真正驻留四CTA，run32约150主要含reduction展开rename，因此缩短stage与展开同时缩短live range；如果仍>128，就没有跨档收益，不采用maxrreg强制spill。代价是组内stage屏障翻倍与loop控制增多。此为资源边界驱动，不能按occupancy上升本身判断性能。

run37为30.192380 ms对照36.179916 ms，134/145register、零spill，仍只有三个CTA/SM；16项数值通过。run38只将MMA内循环展开由2改为1，试验是否消除rename并跨越≤128register边界。预期减少静态fetch但增加每64FFMA后的循环控制；失败则恢复较长stage，不能继续只追求寄存器数量。

run38达到120/123register、零spill、四CTA/16warp资源条件，16项数值通过，但29.162393 ms仍退化。run39仅将stage恢复8，MMA展开保持1，shared恢复29.75 KiB/三个CTA；与run32比较可隔离短指令主体对fetch的影响，与run38比较可隔离屏障减少的影响。矩阵工作量、所有权与统一CTA同步保持不变。

run39为30.076929 ms对照36.074084 ms，124/127register、零spill、数值通过，短主体不采用。

## 1.24 官方arrive/sync双向协议

run40恢复stage8/展开4、分离角色循环，但跨角色用[NVIDIA PTX官方producer/consumer示例](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)的`bar.arrive`与`bar.sync`配对。consumer启动先向空buffer的消费完成barrier3 arrive；producer在QK/softmax完成后sync3等待前P消费，再写P/alpha并arrive4发布；consumer sync4后消费再arrive3。两组各64thread，跨组count128，组内stage仍ID1/2count64。producer在最后发布后再sync3，保持其warp存活直到最终buffer被消费，禁止提前exit导致参与计数变化。

此协议可让producer先计算下一QK、consumer同时计算前PV，只有实际buffer依赖发生等待；不再让两组在分歧PC上都执行相同sync指令。PTX说明不同warp可以执行同ID/count的arrive/sync形式，也要求在barrier reset前不得重复arrival；单buffer完整握手保证一轮一arrival。文档同时说明sm6x/bar.sync的aligned限制，是否被sanitizer接受仍以实测决定，不能凭示例豁免synccheck。

run40数值16项通过、154/159register零spill，26.187546 ms对照36.093024 ms；synccheck插桩后长时间停留在首forward，超过60秒仍未完成，已终止，没有通过证据，不能采用。

## 1.25 交替CTA角色避免scheduler相位偏置

run41回到run32统一循环/共同屏障，仅让奇数CTA交换producer和consumer的物理warp组。逻辑tid仍tid%64，所有QK/PV共享地址和8×8布局保持。依据[Maxwell官方调度结构](https://developer.nvidia.com/blog/5-things-you-should-know-about-new-maxwell-gpu-architecture/)及[Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/pascal-tuning-guide/index.html#instruction-scheduling)，每scheduler绑定独立FP32管线；同一CTA前两warp总是producer，后两总是consumer，三CTA驻留可能让各scheduler上的工作阶段同相，软件流水线无法跨scheduler共享FP32管线。奇偶CTA反转候选可让不同CTA的QK/SFU与PV交错覆盖管线。

warp实际到scheduler的编号映射与CTA到SM顺序未被该官方来源固定，属于待实测调度假设，不能预设奇偶CTA一定同SM。只改变角色映射，资源保持29.75 KiB与约150register；多shape数值和synccheck仍需验证。如果获得收益再考虑与指数FMA组合，避免本轮同时改变算术。

run41为27.875122 ms对照36.094507 ms，150/151register零spill，16项数值通过，角色反转没有净收益。

## 1.26 统一MMA主体以减少指令取指

run42将producer QK和consumer PV每stage的8×8外积放到同一处MMA调用，runtime shared指针选择Q/P和K/V，A stride选择68/64，V改为68stride以与K共享读取主体。两个角色仍分别持有score/output accumulator；所有线程共同经过stage屏障，QK与PV每轮同为64reduction，因此可一一对齐。softmax与新P写仍在consumer完成前P后执行，单buffer不存在并发覆写。shared总30720 B=30 KiB、预计三CTA/SM，global scratch仍8 MiB，不生成S×S。

run31诊断fetch15.75%、exec19.45%、sync24.38%，统一主体针对fetch重复但不消除sync；每stage runtime指针与stride和不同角色分支增加地址开销，而且共同stage屏障使两组必须逐stage对齐，是明确代价。MMA静态代码是否真正合并需SASS检查，不能仅看C++调用一次。数值先验收再计时，若慢则不采用。

run42为32.646143 ms对照36.063437 ms，174/175register、30 KiB shared零spill，只可驻留两个CTA，16项数值通过；统一主体退化，不采用。

## 1.27 QK末stage屏障与maximum判断合并

run43恢复run36普通两warp路径，仅把QK末stage的`__syncthreads`省掉，因为紧接的`__syncthreads_or`在任何union覆写之前已经使所有线程到达并完成QK shared读取。stage0..6的发布屏障保持，末stage不发布下stage；期间score mask/局部maximum仅访问register。PV末stage屏障明确移到循环之后，仍在下一tile复用union之前执行，没有减少PV保护。每tile净省一次CTA屏障，但保留条件maximum的原统计语义；必须racecheck/synccheck验证后才能采用。

run43为26.645557 ms对照36.041266 ms，16项数值通过，小幅改善仍未达标。

## 1.28 打包后移除shared padding

run44仅将run43的Q/K/P shared stride68改64。run14的相同padding移除在row-major输入与scalar shared转置时退化，不能直接外推到run19后的packed/vector store：本路径Q/K每stage的所有global与shared写均为四元素连续、row64，padding不再用于scalar转置的避冲突。固定dimension的MMA读取bank集合不变，stride64允许地址移位，shared从22 KiB降至21 KiB，但四CTA驻留上限不变。需要实测净收益而不能因除去padding就断言无冲突；P store/vector quad映射也必须synccheck/racecheck验收。

run44为26.545554 ms对照36.032407 ms，223/218register零spill、16项数值通过。

## 1.29 两warp普通CTA的stage4资源边界

run45保持run44普通64thread和每线程8×8，仅stage8变4、MMA展开4变2。shared由21 KiB降到19 KiB（P16 KiB、双V2 KiB、统计1 KiB），96 KiB/SM可容纳五CTA；寄存器必须≤200/线程才能满足五CTA的64K register预算及分配粒度。run44约220主要含128accumulator与reduction rename，短stage/global fragments和展开可能降低约20float，但需ptxas实测。相比角色流水线stage4，该方案不需要跨组握手、仍按QK→softmax→PV顺序，代价为每阶段屏障翻倍。只有真实延迟改善才能采用，资源跨档本身不是结果。

run45为34.201239 ms对照36.083713 ms，203/202register、零spill、16项数值通过，未跨200register边界。run46仅MMA展开2变1，检验更短live range能否跨档；run38角色流水线跨128后的收益不足不直接代表普通CTA，但这是本资源边界的最后一次单变量检验，失败则恢复stage8。

run46达到198/194register但40.528485 ms退化，16项数值通过，跨驻留边界不足以补偿循环/屏障，恢复stage8。

## 1.30 屏障前预取最后fragment、屏障后计算

run47在普通64thread、stage8、stride64与条件基准上，依据[MaxAs最后FFMA块前放BAR.SYNC](https://github.com/NervanaSystems/maxas/wiki/SGEMM#calculating-c-register-banks-and-reuse)调整stage边界。每stage先计算前7dimension，在register读取最后dimension的Q/K或P/V，然后发布下一shared stage并执行原屏障，最后以已加载fragment计算最后64FFMA。这样屏障保证所有前shared读取完成，下一stage可复用另一buffer，同时屏障释放后立即有64个独立FFMA，随后才读取下一shared数据。FP32 reduction顺序仍0..7，不移动同accumulator的加法。

QK最后stage仍由随后的CTA条件判断保护union；PV最后stage仍用tile末屏障保护下一tile复用。增加末fragment16float live range与独立FFMA code body是代价；展开前7步可能改变ptxas调度，因此需资源、数值与实际时长验证，不把社区SGEMM近98%吞吐当本算子的预期。

run47为31.963831 ms对照36.080896 ms，218/216register零spill、16项数值通过；stage边界前移未获得收益，恢复普通边界。

## 1.31 顺序执行共享单一MMA主体

run48保持两warp普通CTA与QK→softmax→PV顺序，但由两phase runtime循环执行同一stage/MMA主体。Q/P的A stride64、K/V的B stride68统一，phase0写score、phase1写output；shared指针在phase选择。不同于run42同时producer/consumer，不需要跨组握手或逐角色对齐，shared约21.25 KiB且四CTA仍可能。代价是runtime accumulator选择可能使编译器将本应register的数组spill至local，phase分支与指针增加整数指令；若spill或静态FFMA重复仍存在即否证。保留run44条件软max、所有准确性契约，先检查编译与16项数值再计时。

run48为102.405579 ms对照36.282932 ms，512 B local stack（虽ptxas spill计数为0，address-taken arrays仍进入local），16项数值通过但严重退化。恢复run44式独立MMA主体，不能以共享代码换取local accumulator。

## 1.32 范数上界支持固定softmax平移

run49依据Cauchy–Schwarz不等式`|Q_i·K_j|≤||Q_i||₂||K_j||₂`，在O(Sd)准备kernel中求全局最大Q/K范数与最大|V|。每float平方与累加使用向上舍入，sqrt也向上舍入，scale乘法加入1.0001倍余量覆盖64个有序FP32 FMA的舍入界。取base=ceil(max_norm_Q×max_norm_K×scale×1.0001)。若base≤32且max|V|≤1e20，score位于约[-base,base]，使用全程固定base，不做maximum归约或重缩放；否则保留原条件online softmax。数学上softmax平移不要求基准是实际maximum，分子/分母使用相同base即保持函数值。

eligible输入的P区间至少约[2^-64,1]，指数平移score×scale+(64−base)位于[0,64]，不产生指数underflow；denominator至少有合法key且normal。分子最大S×1e20在32-bit索引范围仍远低于FP32最大值，保守V界避免固定基准之前的大数风险。非有限范数/V映射为∞并回退，次正规概率×大V和大score也回退，不裁剪概率，不降精度。

准备kernel的norm向上归约用16lane处理完整64dimension；global额外统计只有3个32bit，包含清零、准备、pack与主体的端到端计时。主体CTA一致选择fixed/online，固定路径每tile仍有union保护屏障和causal对角mask，所有QK/PV工作量保持。对抵消的FP32舍入误差仍需2e-5完整测试，理论范围界不自动保证相同求和精度。

run49为26.274640 ms对照36.101428 ms，226/219register零spill，16项数值通过。run50在kernel入口CTA一致选择编译期`FIXED_BASE`body，允许ptxas删除fixed路径的全部maximum、rescale和行基准状态，不在tile内保留runtime分支；shared用同一21504 B动态区域，避免两body静态shared叠加。增加静态函数体积与入口分支为代价，register仍由两body最大需求决定；需要SASS/精度/计时验证。

run50为25.948407 ms对照36.090625 ms，224/223register零spill，16项数值通过；固定基准编译期路径有效但距离60%目标仍差约4.3 ms。

run50在完整小/中shape套件下memcheck与synccheck均0 errors、racecheck为0 hazards/0 warnings（15项，不含额外long-sequence抽样项）。主体诊断occupancy12.34%、issue75.18%、exec30.87%、fetch11.17%、sync16.50%、memory dependency0.91%，动态指令1.298626e9。主路径在范围安全输入上已去掉逐tile softmax最大值/重缩放，下一步不能继续凭历史softmax开销扩大优化；需要针对SIMT矩阵指令顺序、register bank与实际stage调度获得更多收益。

## 1.33 原始MaxAs两步swirl依赖图

run51只更换8×8 inline PTX的独立FFMA顺序：采用[MaxAs SGEMM64源码](https://github.com/NervanaSystems/maxas/blob/master/sgemm/sgemm64.sass)中swirl `[2,0],[2,1],[0,1],[0,0]`、x=`0,1,32,33`、y=`0,2,32,34`且相邻y反向x的顺序。映射到本8×8索引后，同一pair的query跨两个register bank候选间距，而列为相邻两个；原run10式pair query相邻。仍保持每个output的dimension累加顺序、RN FMA与所有访存/softmax逻辑。

PTX不能固定SASS register分配或指令顺序，本轮只给ptxas不同依赖图并检查其真实调度与resource。原作者强调swirl与register bank布局的联合收益，本实验不预设仅改PTX就达到该收益；16项数值和实测计时分别验收。

run51为25.861069 ms对照36.118375 ms，223/226register零spill，16项数值通过，差异仍小，需成对重复确认后才能把swirl视作独立收益。

## 1.34 固定基准与合法共同循环流水线

run52结合1.32的范围证明与1.19已通过synccheck的共同角色循环。producer仅持有64float score及8float分母，consumer仅持有64float output；固定基准删除逐tile maximum、shuffle、alpha发布和output重缩放。两组各64thread，QK(t)与PV(t−1)并行，单P缓冲在共同消费完成屏障后才覆写，随后共同发布屏障；不恢复曾报错的分离角色循环。最后stage没有下次组内shared写，不需要组内屏障，下一轮的共同CTA屏障在任何该组shared复用之前保护所有读取。

Q/K/V stride64各双stage8，shared总29184 B（28.5 KiB），96 KiB/SM允许三CTA、12warp；如果编译register≤约168，则寄存器不限制此驻留。相比run50的两warp四CTA、8warp，候选收益来自更短live range和SFU/FFMA重叠；所有warp仍共用FP32管线，不能视为矩阵运算减半。入口按设备范围界先执行固定pipeline kernel，不满足条件时立即返回，再执行单独fallback kernel，防止fallback的较高register需求把pipeline驻留压低。额外kernel启动完整纳入端到端计时，不产生CPU同步。

预期register约130–160，shared28.5 KiB、零local accumulator；必须以编译资源、精度、真实计时及同步检查否证。先单一组合实验，若收益不足，不能只按更高occupancy采用。

run52为28.181691 ms，16项数值通过，161/127register、29184 B shared、零spill，未获得净收益，不采用。

## 1.35 固定基准下的128行CTA

run53采用128query×64key、128thread，每两个warp使用同一MaxAs64×64映射，两个半CTA共同搬运K/V，P为64×128。每线程score/output仍各64float，fixed删除maximum/alpha/行基准，推迟分母归约；Q/K先O(Sd)打包再vector搬运。shared为P32 KiB+V双stage4 KiB+最终分母1 KiB，总37 KiB、最多两个CTA/8warp。global fragment为Q8/K4/V4，stage8，展开4。

run3/run8的大CTA受255register及spill限制，现有fixed路径省去了16行状态与重缩放temporaries，因此预计约210–240register、两个CTA可驻留。本轮检验扩大query带来的K/V复用和CTA数量减半能否在相同8warp驻留下获益；每FFMA operand复用比保持8×8，不能预测计算工作量下降。入口与fallback分离、所有准备及启动都纳入端到端计时，causal对角最后两个64key tile按绝对query比较。若spill或实际变慢则否证，不继续放大相同微矩阵。

run53为25.990876 ms对照35.987763 ms，219/208register、37888 B shared、零spill，16项数值通过，尚未获得净收益。run54只将stage8扩为16，shared变P32 KiB+V8 KiB+分母1 KiB=41 KiB，96 KiB/SM仍两CTA，前提是register≤256的两CTA预算且无spill。相比run21在64行CTA上stage16损失一个CTA，本候选的驻留档位不变，因此仍可独立检验屏障减少；global fragment Q16/K8/V8的额外register为明确代价，展开保持4。

run54为24.622097 ms对照36.006324 ms，246/227register、41984 B shared、零spill、16项数值通过，约68.38%对照耗时，是新的最佳候选。下一轮需要先对该主体诊断，避免以旧run50的计数器指导新资源布局。

run54诊断issue78.76%、sync13.32%、exec33.16%、fetch12.32%、memory0.53%，动态1.280665088e9、IPC3.350991，synccheck全小/中shape套件0 errors。相比run50屏障减少使issue改善，但仍有stage调度与fetch成本。

## 1.36 固定基准的split-K与线性归并

run55将每128query CTA的key区间分成四个独立部分（小S采用两部分，确保64key整块）。依据[CUTLASS split-K并行归约](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/efficient_gemm.html#parallelized-reductions)提高CTA调度粒度，但不借用其GEMM加速数字。传统attention split-K需保存每部分maximum并重缩放；固定全局base使不同部分的P具有同一平移因子，因此直接加未归一化分子和分母，再做一次除法。不同CTA无依赖，最后单独combine kernel以固定split顺序相加，不使用atomic浮点累加。

global新增num为split×Sd、den为split×S，split=4时主形状16 MiB+256 KiB，不是S×S矩阵；全forward包含分配、写回和combine。QK/PV矩阵总工作不变，shared41 KiB与register预算不变；CTA从128增加到512，可减少最后一波不满与causal负载不均，不能指望拆分本身降低矩阵指令数。causal无合法key的part显式写0分子/分母，最终只在所有部分求和后除法，避免0/0。分块改变FP32累加顺序，必须2e-5、次正规贡献与抵消全套验证。

run55为23.552868 ms对照36.027460 ms，244/227register、零spill、16项数值通过，约65.37%对照耗时，split-K取得端到端改善。尚差约1.94 ms达到60%，需要降低主体非FFMA而不是把历史目标放宽。

## 1.37 有界固定指数的normal路径

run56仅在eligible fixed路径去掉1.11的`x+64`和结果`×2^-64`，指数直接`ex2.approx.ftz(fma(score,scale,-base))`。1.32已证明base≤32时合法score−base≥−64，全部P为normal FP32；masked −∞仍为0，因此FTZ不会丢失有效概率。fallback保持保留subnormal的范围变换指数，极小概率×大V案例走fallback。直接指数每score少一条FMUL，bias为整数负base同时降低仿射平移舍入，矩阵与split归并不变。

预期主形状每线程256tile×64score减少约1/64个FP32 FMUL比例，收益受SFU和FFMA调度制约，不能声称自动补齐1.94 ms。仅凭有界输入删normal保护，而不是全局fast_math；必须全套2e-5及实际计时验证。

run56为23.184107 ms对照36.086788 ms，16项数值通过，约64.25%对照耗时，normal指数路径取得小幅净改善。下一项只将stage16的MMA展开从4改8，检验减少stage内循环控制能否改善约33%的execution dependency；模型代价是静态FFMA主体翻倍及register rename压力。stage与shared仍41 KiB、同一split4，若spill或fetch/延迟退化即恢复4，不继续完整16步展开。

run57展开8退化为26.999664 ms对照36.087025 ms，248/224register、零spill、16项数值通过，恢复展开4。run58只将长S的split4改16，每部分64tile变16tile，CTA从512改2048；额外global num从16 MiB到64 MiB、归并读写及短CTA启动/尾部成本为明确代价。该独立实验来自run55已取得的调度粒度收益，检验更短任务是否进一步降低CTA尾部与causal不均；不把增加CTA当作必然改善，不改变每tile算法和MMA展开。

run58为23.580110 ms对照36.041635 ms，16项数值通过，split16没有净收益，恢复split4。run59在128行fixed/split4路径只加入双register fragment：每dimension先读取下一Q/K或P/V fragment再更新当前64accumulator，循环展开仍4，保持FMA累加顺序。run13曾在64行online路径退化，但本固定路径与大CTA的stage16、较少行状态不同，尚可独立检验。本轮额外16float live range可能超过255并spill，若发生立即否证；不强制maxrreg隐藏成本。理论来源仍CUTLASS/MaxAs的shared→register软件流水线，主要目标是LDS latency与FFMA重叠。

run59为23.029902 ms对照35.998128 ms，242/238register、零spill、16项数值通过，小幅改善，仍约63.98%对照耗时。

## 1.38 可验证的FFMA/LSU双发射控制

run60依据[Pascal指令调度官方说明](https://docs.nvidia.com/cuda/archive/12.8.0/pascal-tuning-guide/index.html#instruction-scheduling)的算术与memory双发射能力、[MaxAs SGEMM64](https://github.com/NervanaSystems/maxas/blob/master/sgemm/sgemm64.sass)在FFMA后LDS使用stall0的范例，仅将相邻独立FFMA→LDS/LDG前者的stall1改0。不移动instruction word、不改reuse/yield/scoreboard/branch。必须校验FFMA destination不是下一load的地址或destination quad，两个slot没有waitbarrier，FFMA没有read/write scoreboard；其余控制保持。

这不是消除真正依赖等待，而是允许不同管线同cycle发射，硬件若不支持某配对将决定实际效果。所有原有load scoreboards与消费者waitmask保持，FFMA固定延迟靠后续数值、sanitizer进一步验证；若任何精度失败或无端到端收益则回退标准编译。前后extension/cubin哈希及每个改动地址均归档，不能只以源码哈希宣称可复现。当前SASS显示若干潜在配对，预期改善有限，不把社区SGEMM吞吐直接迁移到attention。

run60的保守依赖筛选未找到可修改配对，changed bytes=0；23.053831 ms对照36.042258 ms仅属于run59标准二进制再次计时，不构成双发射优化。不能为了命中配对放宽真实依赖条件，当前最佳仍run59约23.03 ms。

## 1.39 16×8微矩阵与固定基准流水线的资源证明

下一项依据CUTLASS/MaxAs的register tiling扩大每角色微矩阵，而不是仅增加stage/CTA。producer两warp处理128query×64key，每线程16×8 score；consumer两warp处理同128query×64output，每线程16×8 output。分离角色使每线程只持有一个128float accumulator（普通顺序kernel则需256float，超过255硬上限），固定基准无maximum/alpha，不额外保存第二矩阵。每dimension operand由16float/64FFMA变成24float/128FFMA，比例从0.25降0.1875；LDS.128比例由4/64变6/128，减少25%主要shared operand指令，所有数学FMA工作不变。

使用stage8、stride128的Q双buffer8 KiB、K双buffer4 KiB、P32 KiB、V双buffer4 KiB，总恰48 KiB/CTA；96 KiB/SM可驻留两个CTA、8warp，与当前fixed_wide相同。最终分母2×128float借用Q区域：producer最后QK完成后先组内同步，再写分母，后续无Q复用，consumer仅读P/V；共同CTA屏障发布分母。没有增加静态shared超过硬限制，没有全局S×S。若寄存器≤256且零spill则两CTA可驻留，但最终以ptxas资源验收。

共同循环与两个共同CTA屏障沿用run32合法协议。producer QK(t)和consumer PV(t−1)并行，单P缓冲在消费完成前不得写。可沿用split4增加query/keys任务粒度，并以同base线性归并。预期score/output128+operands24+global Q16/K8/V8+den16及地址状态接近200–240register，若spill或性能退化即否证。此前run52的8×8角色流水线慢不能直接否证本25%operand比例改善，但同步和取指成本仍需实际检查。

run61为25.879496 ms对照36.033512 ms，255/236register、48 KiB shared、零spill、16项数值通过；没有净收益，预估register再次低估编译rename。run62仅MMA展开2变4，依据先前stage-loop控制与register复用的实测差异做该微矩阵的最后检验：每dimension128FFMA保持，静态主体增大，若spill或仍慢就结束该方案并回到run59，不进一步堆叠参数。

run62为36.000464 ms对照36.096952 ms，247/245register、零spill、16项数值通过，16×8角色微矩阵方案不采用。恢复run59标准源码/编译（run63复现及剖析），不应用run60控制位patch。

## 1.40 大CTA的完整reduction shared tile

run63复现约23 ms的最佳路径，synccheck为0 errors；三seed、每seed五轮交替顺序的non-causal耗时比约63.8–64.4%，causal约63.0–64.4%，仍未达到60%。剖析issue80.84%、sync12.47%、exec29.77%、fetch10.54%、memory0.34%，证明64%附近效果可重复，而非单轮噪声。继续以资源模型降低stage同步。

run64在128query CTA中完整载入Q128×64及K64×64到shared，共48 KiB；PV也完整载入P64×128及V64×64，仍48 KiB，两阶段使用union。最终分母借用QK union的首1 KiB，所有PV读取已由tile末屏障保护；无需增加shared到49 KiB。双CTA/8warp驻留保持，普通每线程8×8 score/output不变。搬运用float4逐小组直接global→shared，片段只保留4float并在MMA前结束生命周期，避免全Q64/K32元素同时保持global register。Q沿用64row packed布局，K与V连续直接拷贝。

QK/PV reduction各完整64步、只展开4，不再stage8/16间屏障与global预取。每tile只保留QK发布、union复用、PV发布、PV结束四次共同CTA屏障，降低同步/地址控制；代价是取消global-stage预取、加载和计算顺序执行，仍可能memory latency上升。此前run26持久Q在64query两warp下只剩四warp驻留且退化，本设计128query四warp保持八warp，不持久保留Q，union允许完整两个矩阵各48 KiB。两者不能直接等价。需要resource、精度、端到端时长及sanitizer否证。

run64为28.473029 ms对照36.040622 ms，179/176register、48 KiB shared、零spill、16项数值通过；完整tile顺序搬运未取得收益，恢复run63/59 stage16双fragment。run65只使用官方ptxas `--register-usage-level=0`检验该新主体的rename/取指压力：run23旧两warponline主体无收益不能代表当前四warpfixed/split主体，当前寄存器242/238与大MMA活跃范围提供独立依据。所有算法、stage、split、shared与源码保持；若无收益恢复默认，编译参数及binary哈希归档。

run65为23.165419 ms对照36.040601 ms，242/238register、零spill、16项数值通过，寄存器优化级别未取得收益，恢复默认。

## 1.41 首dimension乘法直接初始化score

run66针对run63 SASS中每tile64条`MOV Rscore,RZ`，令首dimension的score直接以`Q0×K0`初始化，后63dimension继续RN FMA。FP32 `fma(a,b,+0)`与RN乘法在有限输入下相同舍入（signed zero可不同但后续attention无可见影响），不改dot积次序，不降低精度。PV首dimension仍累加已有output，因此不可无条件同样初始化；只有首QK stage调用initialize，其他stage和PV保留FMA。

候选每tile省64条score清零，但首dimension从原统一MMA主体分离可能增加静态fetch和分支，stage余15步展开4的尾部也可能恶化代码生成。必须实际resource、精度和时长否证；不把少MOV当作目标已达成。

run66为24.060005 ms对照36.051609 ms，两个模板255register、零spill、16项数值通过，直接初始化未获净收益，恢复run63主体。

## 1.42 保留global预取的stage32边界

run67与run64完整顺序搬运不同，保留双stage global预取，将run63 stage16扩32：Q/K双stage共2×32×(128+64)×4=48 KiB；P32 KiB+V双stage16 KiB同样48 KiB。最终分母借用已完成PV读取后的union首1 KiB，恰好满足48 KiB/CTA，仍可两CTA八warp。每QK/PV只两个stage，stage间CTA屏障较run63再减半，代价是Qglobal32+K16/V16float live range增加，可能255register/spill。

这是当前大CTA在48 KiB硬边界内保持预取的最大stage；run64取消预取退化不能否证该设计，但若编译spill或实际退化就恢复stage16。MMA双register fragment、展开4、split4与normal指数保持，矩阵工作与准确性契约不变。

run67为23.484548 ms对照36.047990 ms，255register、104/80 B local stack、非零spill，16项数值通过，资源假设失败。run68仅取消register双fragment，保留stage32 global预取，预期省16float live range以消除spill；run59的双fragment收益仅约0.15 ms，不足以 justify stage32 spill。展开4、shared48 KiB与所有数学逻辑保持。如果仍spill或没有净收益则恢复stage16，不通过maxrreg把真实成本掩盖。

run68为24.140785 ms对照36.064041 ms，255register、80/96 B local stack、非零spill、16项数值通过，stage32不采用，恢复run63的stage16双fragment与41 KiB shared。

## 1.43 P共享存储的四元素置换

run69依据V4已有的probability quad置换与MaxAs shared广播映射，给128行P每key的query quad施加XOR偏移`row ^ ((key/4)%8)*4`。完整tile内每key仅置换row低5bit，不跨64行half，不改概率值。producer vector store四元素保持对齐，consumer A fragment分别按key dimension恢复对应quad。Q/K读取不改布局；只PV使用置换A，因此在MMA模板用编译期开关选择offset，不影响QK。

run63 shared store transactions/request5.32包含128-bit基础事务，不能直接称为5.32路bank冲突；该实验检验P跨线程store与consumer广播集合的bank映射。置换增加少量地址XOR/shift，若时长无净改善即恢复。前几轮padding/XOR结果属于不同P stride、微矩阵或global搬运布局，当前128行packed path尚无该单变量消融。

run69为23.495970 ms对照35.991854 ms，245/238register、零spill、16项数值通过，P置换没有净收益，恢复原P布局。run70只把每行八P的顺序分母求和改成二叉加法树；run63 SASS逐行链式FADD限制softmax片段的依赖长度，八项由七级链变三级树，并消除从+0开始的冗余加法。P本身与PV的逐key累加顺序不变，分母的FP32求和次序改变，因此必须保持2e-5，仍使用最终跨warp/跨split线性归并。该优化不使用approx reciprocal或近似softmax，收益若无则回退。

run70为23.232792 ms对照36.066106 ms，16项数值通过，分母树无净收益，恢复顺序求和。

## 1.44 PTX原生CTA屏障与附加MEMBAR

run71针对run63 SASS的每次`BAR.SYNC`后紧跟`MEMBAR.CTA`，将主fixed_wide的`__syncthreads`改成共同控制位置的`asm volatile("bar.sync 0;" ::: "memory")`。依据[NVIDIA PTX bar官方语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)，bar.sync完成保证此前memory访问对参与线程可见，且此thread的新memory访问不得在barrier完成前请求；该指令本身足以发布shared，memory clobber阻止编译器跨屏障重排。所有线程均在共同同一PC执行，满足sm6x aligned/convergence约束。

本kernel的CTA线程仅通过shared通信，global Q/K/V只读，partial输出各thread互斥且kernel结束后交combine，因此无需跨CTA的附加fence。本轮检查原生PTX是否使编译器不再发多余MEMBAR而净降低同步成本；若编译仍附加则没有删除证据。必须SASS、数值、racecheck和synccheck验证，不能因为语义推导就豁免检测。fallback/通用kernel维持现有同步。

run71为23.197610 ms对照36.037852 ms，16项数值通过；SASS仍7条BAR.SYNC和7条MEMBAR.CTA，没有附加fence删除证据，不采用，恢复run63原同步。

## 1.45 纯算术寄存器bank重分配

run72依据[MaxAs C寄存器bank布局](https://github.com/NervanaSystems/maxas/wiki/SGEMM#calculating-c-register-banks-and-reuse)及[CuAssembler SM61编码模型](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuSMVersion.py)，仅对完整kernel生命周期中只用于已解码scalar算术指令的register做一一置换。所有出现在load/store、地址、整数、shuffle、分支等指令中的register及其所属完整四寄存器quad均冻结，避免隐式128-bit destination或64-bit地址pair被破坏。FFMA、FADD、FMUL、MOV、RRO.EX2、MUFU.EX2的field在修改前必须逐operand与disassembly核对；遇到未识别编码直接失败，不能凭低位猜测继续。

按四bank和原reuse bit模拟未命中operand读取，计算每候选scalar register落在各bank与固定A/B operand冲突次数。以Hungarian O(n³)确定最小代价双射，只使用原register集合，register count、所有指令顺序、control fields、依赖barrier及ELF metadata保持。模型只是社区来源假设，真实operand collector未完全建模；不把模型cost直接当运行时收益。

全kernel一一rename保存所有值的生命周期与依赖图，数值算术顺序不变。需归档每个register映射、前后cubin/extension哈希、实际改动instruction数量；先16项精度再端到端计时。若无收益即恢复标准编译，不把二进制优化悄悄作为源码结果。

run72离线检查没有找到模型收益：两个模板的可改scalar映射均保持identity、changed bytes=0，未启动GPU计时，该编号仅归档理论否证。原reuse有效后，单可改C source与固定A/B bank没有本模型可减少的冲突；不能为得到改动而忽略reuse或解冻隐式向量register。

## 1.46 8×8外积的对偶操作数顺序

run73只交换inline PTX每条RN FMA的乘法A/B operand，accumulator和每元素reduction顺序完全保持。有限FP32乘法交换不会改变精度；不同source slot reuse-cache各自独立，ptxas物理register/bank分配及遍历调度会随PTX输入依赖图改变。依据CUTLASS SIMT外积与MaxAs独立source-slot reuse，两操作数没有必须固定为Q/K顺序的数学限制，但该编码仅是调度候选，不能提前预测收益。shared/global布局、stage16、split4、固定指数和所有屏障保持；须resource、SASS和实际时长验证。

run73为23.027382 ms对照36.009820 ms，242/238register、零spill、16项数值通过，没有独立净收益，恢复原operand顺序。

## 1.47 寄存器保留128key，分两半发布P

run74依据CUTLASS/MaxAs的register tiling，将普通128thread CTA从128query×64key改成128query×128key。每线程QK为8×16，PV仍为8×8：每dimension读取8个Q与16个K，执行128FFMA，operand/FFMA由16/64降至24/128，QK部分的shared operand指令减少25%。矩阵工作量不变，QK循环及Q搬运次数减半。

完整P128×128需要64 KiB，超过Pascal每CTA48 KiB限制，因此128个score先保留在register；先发布前64key的P并执行PV，再发布后64key的P并执行PV。每个PV半块沿用32 KiB P加双V stage，最大shared约37 KiB，仍可两CTA八warp。QK双stage8的Q/K共16 KiB，同一union复用。causal逐元素使用绝对key位置屏蔽，split区间保证128key对齐。

score128、output64、分母8、global16、单fragment24及地址状态接近255register上限；本轮先用单fragment和展开2控制rename，而不假设8×16必然更快。两个8×8外积共享同一个Q fragment，保持RN FMA及每元素的dimension/key顺序；P两次发布增加union保护屏障，是明确代价。如果spill或耗时没有净收益即否证，不以操作数模型替代实际资源和精度检查。

run74为25.832904 ms对照36.063232 ms，两个模板255register且32/24 B stack、非零spill，16项数值通过。资源模型失败。run75仅将QK两次8×8 PTX外积合并为一个8×16 PTX依赖图；128个score同时显式作为输入/输出，Q8与K16各一次输入，避免两个asm边界的accumulator rename。仍展开2、stage8、P分半和所有同步不变；这不承诺ptxas固定register，若仍spill或无收益则终止本大key方案。

run75为25.845953 ms对照36.150040 ms，资源与spill完全相同，16项数值通过。结束128key微矩阵方向，恢复run63。run76针对run63双fragment SASS中仍存在的BFE与fragment奇偶选择，将单dimension循环改为显式成对迭代：固定fragment0执行偶数dimension、fragment1执行奇数dimension，在各自FFMA前加载下一片段，循环每次前进2dimension且展开2对。静态展开仍总4dimension，shared/global stage16与所有算术次序不变，消除runtime `%2`索引而保留prefetch。理论依据为编译期地址与所有权消除非矩阵控制；需以实际SASS、resource与延迟检查是否真正减少指令。

run76为23.005437 ms对照36.067712 ms，242/238register、零spill、16项数值通过，资源与run63完全相同，约0.015 ms差异不构成独立收益。

## 1.48 完整K/V与跨tile global预取

run77保留128query×64key、普通128thread、8×8微矩阵与Q双stage16，却将K/V分别完整存入shared。QK shared为Q双stage16 KiB加K16 KiB=32 KiB；PV为P32 KiB加V16 KiB=48 KiB，两阶段union，最终分母借用已完成PV的union首1 KiB。仍两CTA八warp，Q阶段仍四次发布，PV reduction64步无stage屏障。

与run64完整tile直接搬运不同，本轮在PV(t)开始前global预取K(t+1)到32float/thread register，在PV64步期间隐藏延迟，再于tile结束屏障后发布至shared；初始K单独准备。V(t)在最后QK stage计算前预取32float/thread，QK剩余16dimension隐藏其延迟，QK结束屏障后才写V与P。下一Q首stage同时在PV前预取16float/thread。QK中stage预取保持；不在同一时刻持有K与V两个global完整片段，已存shared的V不再作为register使用。

候选删除每tile三次PV stage屏障及对应V stage地址，但新增32float完整global live range。score/output、双MMA fragment与Q/V片段可能造成超过255register，必须ptxas验收。所有shared union覆写都位于共同CTA屏障之后，K写完在下一Q发布屏障后读取，不跨warp分离barrier，不减矩阵工作和精度。若spill或端到端无收益即否证。

run77为22.189722 ms对照36.029114 ms，254/242register、48 KiB shared、零spill，16项数值通过且synccheck0 errors。诊断issue83.68%、exec24.68%、fetch10.93%、sync9.90%、memory0.085%，相比run63减少同步并改善issue。新的61.59%结果仍未达到60%。

run78仅将完整PV64dimension的四次stage16 MMA调用合成一条连续64dimension双fragment循环，展开仍2对/4dimension。因为V/P完整存shared且无中间发布，PV不需要stage边界的末fragment分支、重新加载第一fragment或outer-stage loop；QK仍原stage16。shared与跨tile global预取保持，FP32逐key累加顺序不变。预期减少循环/地址并延续fragment预取，但末尾live range及ptxas调度可能改变，必须独立resource和计时。

run78为22.191188 ms对照36.085965 ms，254/242register、零spill、16项数值通过，无独立净收益。run79只把Q shared stage16改32：Q双buffer32 KiB加完整K16 KiB恰48 KiB，PV仍48 KiB，两CTA不变。两个QK stage使用明确顺序：预取第二Q32→计算第一stage→发布Q→预取V32→计算第二stage。Qglobal在发布第二stage后不再读取，Vglobal到此才写入，明确生命周期避免同时持有32Q+32V；PV后半期间仍预取nextK32及nextQ32。

该候选每tile再省两次Q stage屏障，并让V读取与32dimension的QK计算重叠。global Q片段增16float、static QK两个body增加fetch，是明确代价。不同于run67的Q/K双stage32，本轮K完整shared、V晚于Q发布而没有Kstage global片段，资源推导不同；仍需检查255上限/spill及实测，不能把历史失败或当前48 KiB恰好满足当作性能结论。

run79为22.148022 ms对照36.134808 ms，254/255register、零spill、16项数值通过，约61.29%对照耗时，仍差0.467 ms达到60%。run80只把完整PV双fragment循环从展开2对/4dimension变展开4对/8dimension，QK仍展开2对。PV没有stage barrier且nextK/nextQ的global等待需要与连续计算重叠，独立于旧run57同时扩大QK/PV且逐stage16的body。当前fetch约11%、register255，增加静态body/rename的风险明确；若spill或变慢即恢复，不以循环减少断言收益。

run80为23.263153 ms对照36.066354 ms，causal模板8 B spill、16项数值通过，恢复run79展开4dimension。

## 1.49 split与SM波次数的整除

run81仅把长S的split从4改5，key边界按64key整块计算`start=floor(part×tiles/splits)`与`end=floor((part+1)×tiles/splits)`，确保每key恰被一个part覆盖。主形状有128query CTA组，本卡10SM且每SM两CTA，split4产生512CTA、26波最后12CTA；split5产生640CTA、32个完整20CTA波。各part含51或52个64key tile，最多约2%工作差异。模型希望减少最后不满波，但不能忽略动态调度、shared发布及53%左右的causal负载不均。

新增global partial由16 MiB增20 MiB、den及归并read同比增25%，所有成本计入端到端。矩阵工作/精度保持，split求和顺序改变，仍需16项及FP64校验。该参数来自SM驻留与query分块整除，区别于run58不考虑波次的split16；若无净收益则恢复4，不开展无界参数扫描。

run81为22.125751 ms对照36.037469 ms，254/255register、零spill、16项数值通过。split5较4差异很小，仅可作为调度候选而非确定独立收益。

run82保持完整K跨tile预取，但将nextQ首32dimension的读取后移到PV结束之后，再进入下一tile。此前PV期间同时保持nextK32和nextQ32，额外64float global fragment限制rename；Q工作集只有本CTA128×64、与K相比重复访问相同数据，更可能由L2命中。代价是失去Q读取与PV重叠，收益候选是少32float live range和更好的寄存器调度。Q后32仍在前32的QK前预取、V仍在后32的QK前预取，48 KiB shared/8warp驻留不变。本轮仅移动Q读取位置、保持PV展开4dimension，以检查资源而不同时加展开。

run82为22.029977 ms对照36.128564 ms，254/254register、零spill、16项数值通过，61.00%对照耗时，改善小于原先预估寄存器降幅，仍差约0.353 ms达到目标。

run83只将两个显式QK stage32调用改为一个不展开的两stage循环，使用stage索引选择Q buffer及完整K的起点。stage0预取Q、stage1预取V，仍在各自完整32dimension计算后发布，保持run79生命周期和run82的nextQ后移。两个显式MMA主体使QK静态FFMA体积翻倍，而run77 fetch约11%，因此候选减少取指体积；代价是两stage间的少量循环和runtime地址。每线程score/output仍register，不像run48把不同phase accumulator作为runtime变量选取，不引入address-taken accumulator。

run83为22.321332 ms对照36.099277 ms，255register且24/32 B stack、非零spill，16项数值通过，恢复显式QK阶段。

## 1.50 完整Q与保留K/V预取

run84改为完整Q128×64与完整K64×64 shared、仍48 KiB union。Q在每tile PV结束屏障后以float4小片段直接global→shared，K(t+1)保持在PV(t)前预取且tile末发布；V(t)在QK64步前预取32float/thread。两次QKstage32变一条连续64步双fragment循环、展开4dimension，无中间CTA屏障。

区别于run64：K global latency仍与前一PV64步重叠，V global latency仍与QK64步重叠，只有反复使用同一query工作集的Q不预取；run82已显示取消跨tile Q预取小幅改善，支持检查不保留Qglobal片段的资源收益。本轮Qglobal仅4float瞬时小组、QK需score64+output64+Vglobal32+MMA32及地址，预期比254降低，但不预设数字。PV期间仅保留nextK32、与run82一致。FP32累加次序、split5、normal指数及最终归并保持；若无净收益恢复run82。

run84为23.198105 ms对照36.103204 ms，255register、112 B stack及非零spill，16项数值通过，编译资源假设失败。run85只把copy_q改成局部PTX `ld.global.v4.f32`立即`st.shared.v4.f32`：四float临时值限制在asm内，shared地址用`__cvta_generic_to_shared`，volatile与memory clobber保留每次拷贝的顺序并阻止跨拷贝提前global读取。仍由外层共同CTA屏障发布，PTX拷贝不替代同步。不改变Q FP32 bit、stage或MMA；检查是否消除编译器把16组global Q读取同时保留的机会，而不能靠源码scope宣称只占4register。

run85为21.927673 ms对照36.074582 ms，235/233register、48 KiB shared、零spill、16项数值通过，约60.78%对照耗时，显式小批拷贝消除了run84 spill。仍差约0.283 ms达到60%。

run86将copy_q局部PTX的批次从4float扩大16float，先四条独立`ld.global.v4`再四条`st.shared.v4`，每批跨度为连续4dimension×64row。Qglobal临时值仍不跨batch，12个额外register预计落在235→247范围并保持两CTA，目标是提高copy阶段的memory-level parallelism，避免每次LD立即由STS依赖串行。shared/global原bit与映射保持，四vector destination均对齐，memory clobber仍在每batch边界，共同发布屏障不改；资源和实际时长必须检验。

run86为22.185728 ms对照36.131686 ms，236/234register、零spill、16项数值通过，拷贝批次扩大无收益，恢复run85的4float立即搬运。

run87只将run85的PV完整64步MMA展开4dimension改8dimension，QK仍4dimension。run80的相同PV展开在双stageQ32的255register主体出现spill并退化，而本主体经显式Q拷贝后仅235/233register，约20register余量允许独立检查更大的PV指令块。共享/搬运/split与数学顺序不变，静态fetch增加仍可能退化；若资源溢出或实际时长无收益即恢复。

run87为22.437321 ms对照36.151706 ms，236/240register、零spill、16项数值通过，扩大PV展开仍无收益，恢复4dimension。

run88仅将copy_q的局部PTX `ld.global.v4`改为`ld.global.nc.v4`，通过[NVIDIA PTX只读缓存说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc)使用非一致只读cache。Q来自前序同stream pack kernel且主体执行期间完全不变，输出/partial均新分配，不与Q alias，因此不需要读写一致性；准备kernel结束至主体启动的可见性由stream顺序保证。该cache路径可能改善重复Q的命中，但官方也说明延迟可能更长，故只以端到端验证判断。所有拷贝地址、四元素bit、shared发布屏障及MMA保持。

run88为21.975552 ms对照36.116376 ms，235/233register、零spill、16项数值通过，synccheck0 errors，无缓存路径独立收益。诊断issue83.36%、fetch13.46%、exec25.40%、sync6.27%、memory1.06%，动态指令1.24919168e9，说明剩余吞吐仍与指令调度/取指有关。

run89恢复默认Qcache，仅将完整64dimension的QK和PV双fragment循环展开从4dimension缩到2dimension。完整共享tile在64步内连续计算，没有stage16末条件与外层loop，因此旧run20的stage8单fragment及run76的stage16双fragment不能代替本消融。预期静态MMA主体减半、fetch下降与register rename缩短，代价为每128FFMA多一次loop控制/地址。以动态指令、延迟与zero spill否证，不把fetch百分比直接换算绝对收益。

run89为22.785114 ms对照36.170700 ms，238/223register、零spill、16项数值通过，缩小展开无收益，恢复run85的4dimension。

## 1.51 共同softmax平移取零的安全范围

run90保持1.32同一eligible条件`base≤32,Vmax≤1e20`，但fixed指数改为`exp2(score×scale)`，共同平移选0。此时P落在[2^-32,2^32]，均normal；S受32bit索引约束且d64使S≤2^25，因此分母≤2^57、未归一化分子≤2^57×1e20≈1.45e37，低于FP32最大3.4e38。每query至少一个合法key，分母≥2^-32不下溢；split线性归并仍保持同一平移。原fixed的范数界与V界继续保留，范围外走fallback，不剪概率、不降精度。

指数前的RN FMA改成RN乘法，少一个live bias register，缩放舍入不再包含负base抵消，也无需eligible通过后再次读取三bounds并ceil作为主体bias。每score仍一条scalar算术，主要希望更短依赖/更少状态，不预设大收益。与run56去掉保护乘法不同，本轮指数区间上移但仍normal，需长序列、抵消和2e-5数值确认。

run90为21.824136 ms对照36.125375 ms，223/231register、零spill、16项数值通过，60.41%对照耗时，尚差0.149 ms达到60%。run91只将copy_q的16个vector拷贝从完整展开改成每4个展开：局部asm立即load/store不变，不扩大memory batch。当前fetch约13.46%，Q完整拷贝静态16组地址/LDG/STS重复，可减少取指体积；代价为四轮loop控制及address更新。MMA、zero base、split和所有数据bit保持，以实际延迟/resource确认。

run91为22.330151 ms对照36.129126 ms，222/226register、零spill、16项数值通过，缩小copy展开无收益，恢复完整展开。

run92在当前完整Q/K、零基准、223register主体重新检查首dimension直接乘法初始化score。与run66的stage16、255register主体不同，当前完整64步无需首stage runtime选择，可在QK一开始用Q0×K0赋值64score，dimension1及后62步继续RN FMA，避免64条MOV清零。完整shared可连续读0/1、随后从dimension2进入双fragment循环；PV仍从0累加已有output。数学有限输入下RN乘法与RN FMA(+0)同舍入，静态首块和余62步可能改变fetch/rename，因此只有实际精度/资源/延迟能支持采用。

run92为21.917287 ms对照36.106802 ms，230/229register、零spill、16项数值通过，无净收益，恢复run90。

run93仅将主形状split5变10，仍整除本卡20CTA/波，CTA由640增1280、每part由51/52tile缩为25/26tile。该拆分检查更细causal负载调度与最后任务持续时间缩短，partial增加20 MiB、combine读取增加20 MiB及短CTA准备占比提高是代价。旧run58 split16在逐stage主体退化不能直接否证本完整shared/zero-base主体，但不预期单靠CTA数必然提升。所有256key tile恰一次覆盖，float32归并固定顺序且2e-5验证保持；若无净收益恢复5。

run93为21.984129 ms对照36.128134 ms，223/231register、零spill、16项数值通过，split10无收益，恢复5。

run94基于run90只将8×8 PTX外积遍历从MaxAs的跨行swirl改为[CUTLASS SIMT蛇形遍历](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/thread/mma_sm50.h)：以column为外层，偶数column按row0→7、奇数column按row7→0。每条仍RN FMA、每元素dimension顺序保持，两个operand source均原Q/P、K/V。该顺序让同一B连续服务8FFMA，跨相邻column复用端点A，当前235以下register和连续64dimension主体的编译调度可能不同于旧stage主体。PTX不能固定物理bank/reuse，实际资源及SASS为必要证据，时长无收益则恢复MaxAs图。

run94为21.745050 ms对照36.083759 ms，224/230register、零spill、16项数值通过，约60.263%对照耗时，仍差0.095 ms达到60%。run95仅交换该蛇形图的乘法source slot，保持相同指令遍历。B在外层连续复用，可映射至不同reuse-cache slot，实际ptxas分配与控制可能改善；有限FP32乘法对偶不会改变每元素的FMA/reduction顺序。旧run73的MaxAs图且stage16主体无独立收益不能直接外推，仍仅以编译/数值/延迟否证。

run95为21.707300 ms对照36.095129 ms，224/230register、零spill、16项数值通过，约60.139%对照耗时，尚差0.050 ms；小差异必须成对重复确认。

## 1.52 分四query发布P的epilogue

run96把softmax+P store从完整8行分别两遍，改为每4query计算32个指数、更新四个分母、立即写对应8个float4 shared P，再处理另4query。当前SASS把多数64个score缩放连续提前，随后执行RRO/MUFU及P store，缩短P epilogue中score live range有机会减少rename并使SFU/LSU交错。每行八key的分母求和次序保持，QK/PV次序保持；两个quad shared写入完全不重叠，PV仍在同一个发布屏障后读取，不增加同步。此为CUTLASS分片epilogue与FA3指数/数据搬运重叠机制在SIMT的局部实现，不借用Tensor Core/TMA性能数字；实际编译可能仍重排，因此需resource、SASS与时长证明。

run96为21.788928 ms对照36.104755 ms，224/230register、零spill、16项数值通过，四query分片无净收益，恢复run95。

## 1.53 范数上界与Q/K打包合并准备

run97在原32×8 transpose CTA中，Q/K原始value已从global载入shared，每thread持有同一row的4个dimension值；在每次load时额外用向上舍入累加平方。用warp32lane完整row归约，sqrt_ru后每row的lane0 atomicMax全局Q/K范数。只有grid.z的两个dimension-half值不能单独sqrt后取max：因此改为grid.z仅Q/K两个输入，各CTA一次转置完整64dimension的两个32宽half，row squared norm涵盖全部64。Q对应CTA同时读取V同坐标，归约max|V|，其余Q/K位重排仍32×33 tile、两个half顺序复用shared并有屏障。

准备复杂度O(Sd)、global Q/K各读取一次，Q路径额外V一次；删除独立score_bounds prepare kernel及Q/K再次读取，保留bounds清零和主体device eligibility/fallback。新的prepare额外norm/atomic、第二half同步可能改变transpose资源，只有端到端净收益支持采用。RU平方/加法/归约与原16lane的4float×16相同上界机制，树形和顺序不同只改变保守界松紧，不改变eligible范围证明；所有FP32概率/矩阵/归并保持。每row/CTA可并行，跨CTA atomicMax无求和顺序问题。

run97为21.704550 ms对照36.112177 ms，224/230register、零spill、16项数值通过、synccheck0 errors；合并准备没有确定独立收益。三seed各五轮成对计时non-causal median为60.24–60.66%，causal61.26–61.77%，均未达标，需要继续优化主体，不把单轮接近60%当完成。

## 1.54 query组内交错split的CTA顺序

run98仅将CTA launch从grid(query,split)变成grid(split,query)，主体读取`query=blockIdx.y,part=blockIdx.x`，partial存储仍相同part-major。原顺序先遍历128query再换key part，新顺序同query的5part相邻，依据[CUTLASS threadblock rasterization](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/efficient_gemm.html#threadblock-rasterization)改变L2复用距离及causal负载分布；不改变任何CTA内部数学、shared或register映射。代价是相邻CTA不再全读取同一K part，可能失去K/V复用。全part/query笛卡尔积仍恰一次访问，最终输出地址互斥且combine顺序保持。实际CTA调度不保证按x严格顺序，故必须成对性能验证而不是只据launch理论。

run98为21.956541 ms对照36.081046 ms，224/230register、零spill、16项数值通过，交换grid无收益，恢复query最快变化。

run99将完整64dimension的MMA由双fragment预取改成单fragment，仍4dimension展开。当前没有stage边界且每64FFMA有独立accumulator，ptxas可自行在展开的四dimension间安排shared读取；手动双buffer可能引入多余MOV/rename及控制。旧run59双fragment在stage16带来约0.15 ms收益，但完整shared/零基准/蛇形图没有该独立对照。候选每线程少16float operand活跃区间，代价为更短显式LDS latency隐藏；必须实际resource与时长而不是仅寄存器下降验收。

run99为22.066995 ms对照36.068472 ms，225/229register、零spill、16项数值通过，单fragment无收益，恢复run97的双fragment主路径。

## 1.55 最终选定、回退修复与验证

用户在run99后认可当前性能并要求开始总结，本版结束优化。原始目标为两种模式同轮耗时≤60%；该阈值保留为比较参考，不将近似达标改写为每对均严格通过。最终主路径选择run97：完整shared Q/K与P/V union、跨tile K预取、QK前V预取、4float局部PTX Q搬运、固定零指数基准、split5及蛇形PTX依赖图。run98/99未获收益，不进入最终源码。

run100恢复主路径，并把范围外fallback的FA4条件重缩放阈值8移除。fallback严格维护截至当前tile的真实scaled maximum，只要maximum增加即重缩放，防止大V时P可大于1而未归一化分子溢出。新增S128,d64、跨tile score增加5、V=1e35的回归，两种模式均CPU FP64一致。fixed eligibility不变，主形状不会进入该fallback。

run101清理未引用候选头文件、旧prepare kernel与无用模板；归并独立为combine.cuh，主MMA保留原双fragment/展开4dimension，所有失败候选已存在历史快照。正式源码仅保留7个职责文件，标准nvcc/ptxas编译，未使用二进制patch。主路径224/230register、48 KiB shared、零stack/local/spill；最终独立计时21.626163 ms对照36.063272 ms，约59.97%。三个seed每种模式各5轮，non-causal整体耗时比60.43%、causal61.52%，全部30组保留；按用户接受结果结束，不声称所有样本严格低于60%。

最终17项数值通过，16项常规套件分别memcheck/synccheck零errors、racecheck零hazards/零warnings。S16384,d64两种模式各10行FP64抽样最大绝对误差2.384186e-7；48多形状案例全部抽样通过，29项主体及辅助kernel counters、单次时间线和实际扩展哈希归档。独立时长与profiler replay时长分开，详情见[性能总结](perf_v5.md)、[复现文档](build-and-run.md)与[manifest](manifest.json)。
