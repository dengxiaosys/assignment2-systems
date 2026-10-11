# 1. efficient 的资源、指令与源码分析

同一 GTX 1060、FP32、S=16384,d=64、non-causal 下，PyTorch efficient 实际运行的是 **64×64 的 CUTLASS SIMT attention kernel**，4 warp / CTA、168 registers/thread、21.5 KiB dynamic shared。源码确认它采用寄存器矩阵结果、两阶段 matmul、shared union 生命周期复用与 base-2 softmax。V3 借鉴其中的工作组织，实际获得主要收益的组合是寄存器微矩阵复用、64 行 / CTA 与受控展开；直接改 `exp2f` 在本实现中反而退化。

证据来自 [实际时间线](efficient-reference/timeline.csv)、[原始 counters](efficient-reference/counters.csv)、[manifest 中的 efficient_reference.metrics、二进制身份与资源](manifest.json) 和下列 PyTorch v2.6.0 源码。资源可以提示结构，源代码用于确认机制，性能由独立 benchmark 判定。

## 1.1 实际 kernel 身份

### 1.1.1 Launch 与运行资源

| 项目 | efficient 实测 |
|---|---|
| kernel | `fmha_cutlassF_f32_aligned_64x64_rf_sm50` |
| grid | `(256,1,1)` |
| block | `(32,4,1)`，128 threads |
| registers / thread | 168 |
| static shared | 0 B |
| dynamic shared | 21.5 KiB，即 22016 B |
| stack / local | 0 / 0 B |
| 无 counters 单 kernel capture | 39.597281 ms |
| V3 最终当轮无 profiler efficient 均值 | 36.171397 ms |

grid=256 与 16384/64 相符；kernel 模板名与源码的 kQueriesPerBlock/kKeysPerBlock 参数共同支持 64×64 tile。capture 与正式计时来自不同运行，不能拿一次 capture 替代重复计时。

实际 cubin 为 sm_50，在 sm_61 上执行兼容 SIMT 指令。读取的库为 `nanovllm/.../torch/lib/libtorch_cuda.so`，大小 902652937 B，SHA-256 `136a5e4be2aee6e17a38179aeff229f4e0d87c2e43d8ca58e87183df4be73667`；完整路径与工具命令保存于 manifest。库不复制进测量目录。

### 1.1.2 符号与资源提取

`nvprof` 显示的是 demangled 名，cuobjdump 的精确 filter 需要 mangled symbol：

```text
_Z39fmha_cutlassF_f32_aligned_64x64_rf_sm50N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm50ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE
```

用 `nm -D --defined-only` 定位该符号，再执行 `cuobjdump --gpu-architecture sm_50 --function <symbol> --dump-resource-usage/--dump-sass`。资源输出确认 REG=168、STACK=0、LOCAL=0；SHARED=0 仅代表静态声明，动态 shared 必须看 launch，不能误写成该 kernel 没有 shared。

## 1.2 源码能够确认的机制

### 1.2.1 FP32 SIMT，寄存器矩阵块

[gemm_kernel_utils.h 的 DefaultGemmType](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h#L110-L123) 默认采用 `OpClassSimt`、`OpMultiplyAdd`、1×1×1 instruction shape，ThreadK/WarpK 为 8。FP32 Tensor Core 特化要求 arch≥80，当前 `Sm50` 使用默认 SIMT 路径。因而该实现的速度不能归为 GTX 1060 上不存在的 Tensor Core 功能。

[MM0 的配置](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L347-L427) 以 CTA query/key tile 为 ThreadblockShape，WarpShape 为 32×32×WarpK。不同线程持有部分 score fragment，一个线程有多个独立乘加结果；同一 Q/K 操作数可以在 fragment 内服务多个结果，减少每个 FMA 对加载/通信的比例。

[MM1](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L430-L495) 将未归一化 P 存入 shared，作为第二次 P@V 的输入。不是对每个输出行执行独立 scalar key 扫描；它把两个主要部分都组织成矩阵块计算。

### 1.2.2 输出寄存器与 shared union

[kKeepOutputInRF](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L121-L141) 在 kMaxK≤kKeysPerBlock 的 single-value-iteration 情况保持输出 accumulator 在寄存器。当前 head dimension 64、key tile 64 满足该模板条件。寄存器开销用于更多数据与输出 accumulator，并不能只用“寄存器越少越好”评价。

[SharedStorageEpilogueAtEnd](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L523-L570) 用 union 复用 MM0、after-MM0 和 epilogue 的工作区。它根据生命周期而非将所有阶段相加来确定 shared 容量。V3 采用 K 与 P/V union，同时显式建立覆盖前后的 CTA 屏障，不能仅定义 union 就宣布安全。

### 1.2.3 Softmax 的 base-2 路径

[iterative_softmax](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h#L1181-L1300) 在寄存器中处理 score fragment，使用 `exp2f`，旧最大值改变时才 rescale；score 的 scale 含 log2(e) 转换。数学上 \(2^{x\log_2e}=e^x\)，但实际编译、寄存器和分支成本不同。

V3 的 run5 使用相同转换后，通过 FP64 验证，却达到 118.413 ms、178 registers/thread、occupancy 12.3078%，明显比矩阵块的 expf 版本慢。efficient 使用 exp2f 并不意味着任意 kernel 替换指数都会更快。

## 1.3 SASS 与 counters 的辅助判读

### 1.3.1 静态指令

实际符号的 SASS 中有 FFMA、LDS/STS、`MUFU.EX2`，支持 SIMT matmul、shared 访问和 base-2 指数。静态计数为 FFMA 583、SHFL 25、LDS 151、STS 97、BAR.SYNC/MEMBAR.CTA 各 31、MUFU 60，保存于 manifest。

这与 V3 完全展开版的 2304 个 FFMA、受控展开版的 416 个 FFMA 属于不同循环结构，不能按静态指令少就推导速度，或把 31 个屏障位置乘以全网格 tile 数。V3 的 instruction-fetch 变化同时有计时和 counters 支持，才构成减少代码体积的具体依据。

### 1.3.2 低 occupancy 与高 issue 可以共存

| 指标 | efficient | V3 最终 |
|---|---:|---:|
| achieved occupancy | 18.6268% | 23.4428% |
| eligible warps / cycle | 4.324108 | 4.630517 |
| issue slot utilization | 66.997537% | 68.346762% |
| exec dependency | 37.372316% | 58.416421% |
| instruction fetch | 12.554382% | 9.165196% |
| sync | 23.071899% | 3.932961% |
| memory dependency | 0.827313% | 5.328740% |

efficient 只约 12 个 active warp / SM，却有较多就绪工作；V2 的高 occupancy 并未产生同样的 issue。V3 受控展开后提高了就绪指令和 issue，但同轮 efficient 时间仍少约 11.288 ms。不同 kernel 的 stall 分布、执行指令总数和线程工作不同，不能把各百分比按时间相乘做差。

efficient shared load/store transactions per request 为 2.796997/3.435517，V3 为 1/1。这些值还受向量访问宽度影响，不等于 efficient 有“更差 bank 冲突”就应更慢。源码的向量 fragment 操作与总时间需一起判读，不能孤立优化该比值。

## 1.4 仍需探索的差距

V3 最终保留 32-key tile，而 efficient 是 64-key；后者每行递推轮数减半。V3 run4 的直接 64-key 修改导致 d=128 模板超过 48 KiB shared，说明需要先减少 P 工作区或按模板资源重新组织，不能机械放大。

另一个候选是输出/QK 的 lane 布局、向量化加载和小范围预取。V3 的 global store efficiency 50% 表明 8-lane 行组产生分段写入，但只在结束写回 4 MiB 输出，通常不是主循环的最有价值目标。主循环仍应优先评估 operand 布局、可容纳的 key tile 和调度；最终新增优化必须由独立版本测量。

上述源码机制是明确确认项；efficient 每阶段的绝对周期、cache 细分命中和所有寄存器的具体用途没有被本次数据完整恢复，不作推断性定量归因。
