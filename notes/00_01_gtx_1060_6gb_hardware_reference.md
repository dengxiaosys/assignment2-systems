# NVIDIA GeForce GTX 1060 6GB GDDR5 硬件性能参考

## 0. 范围与结论

本文只讨论 NVIDIA 标准桌面版 **GeForce GTX 1060 6GB GDDR5** 的硬件能力，不依赖任何特定 benchmark、框架或本机实测结果。

> 型号警告：GTX 1060 还有同名但硬件配置不同的版本；本文不讨论 3GB 或 GDDR5X 版本，下文所有数值都以标准 6GB GDDR5 参考规格为准。

最重要的结论是：

1. 它是面向游戏图形的 Pascal `GP106` GPU，计算能力为 `6.1`，有 10 个 SM、1280 个 CUDA Core。
2. 标称基础频率为 1506 MHz，Boost 频率为 1708 MHz；按 FP32 FMA 每周期计 2 次浮点操作，理论峰值分别约为 **3.86 TFLOP/s** 和 **4.37 TFLOP/s**。
3. 显存规格为 6 GB GDDR5、192-bit、8 Gbit/s，有效理论带宽为 **192 GB/s**；容量与带宽是两个独立限制。
4. 它没有 Tensor Core，也没有原生 BF16 算术；FP16 算术吞吐只有 FP32 的 $1/64$，FP64 只有 FP32 的 $1/32$，但支持适合量化推理的 INT8 `dp4a`/`dp2a` 指令。
5. 这张卡适合 FP32 图形和通用 CUDA 工作负载，但不是为现代混合精度训练、高 FP64 吞吐或大模型容量设计的计算卡。

## 1. 规格总览

| 项目 | 标准规格 | 来源或说明 |
|---|---:|---|
| GPU / 架构 | GP106 / Pascal | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| 制程 | 16 nm FinFET | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| 晶体管 / 核心面积 | 44 亿 / 200 mm² | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| CUDA Compute Capability | 6.1 | [NVIDIA Legacy CUDA GPU 列表](https://developer.nvidia.com/cuda/gpus/legacy) |
| SM 数量 | 10 | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| CUDA Core 数量 | 1280，即每个 SM 128 个 | 总数来自 [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/)，SM 结构见 [CUDA 11.8 Programming Guide：Compute Capability 6.x](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#architecture-6-x) |
| 基础 / Boost 频率 | 1506 / 1708 MHz | [NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/) |
| 理论 FP32 峰值 | 基础频率约 3.86 TFLOP/s；Boost 频率约 4.37 TFLOP/s | 本文第 3 节推导 |
| 纹理单元 / ROP | 80 / 48 | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| 显存 | 6 GB GDDR5，8 Gbit/s | [NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/) |
| 显存总线 / 理论带宽 | 192-bit / 192 GB/s | [NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/) |
| L2 Cache | 1536 KB，NVIDIA 原始标注 | [NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) |
| 主机总线 | PCI Express 3.0 x16 | [NVIDIA GTX 1060 用户指南](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf) |
| 显卡功率 / 辅助供电 | 120 W / 单个 6-pin | [NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/) |
| 建议系统电源 | 400 W | [NVIDIA GTX 1060 用户指南](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf) |
| 参考版显示输出 | 3 个 DisplayPort、1 个 HDMI、1 个双链路 DVI | [NVIDIA GTX 1060 用户指南](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf) |

## 2. GP106 与 Pascal 计算结构

### 2.1 Pascal 是架构家族，不是一套完全相同的计算配比

GTX 1060 使用 `GP106`，NVIDIA 将其归入 Pascal 架构和 CUDA Compute Capability 6.1。计算能力编号描述硬件可用的指令与资源特征，不是 CUDA Toolkit 的版本号；NVIDIA 的设备表明确把 GeForce GTX 1060 列在 `6.1` 下。[来源：NVIDIA Legacy CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus/legacy)

不能把 Pascal 数据中心芯片 GP100 的参数直接套到 GP106。NVIDIA 的 [Pascal GP100 白皮书](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)说明 GP100 针对 HPC 与深度学习强化了 FP64、FP16、HBM2 和 NVLink；而 [Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html#nvidia-pascal-compute-architecture)明确区分 GP100 与消费级 `6.1` 分支的 SM 和数值吞吐设计。GTX 1060 的硬件判断应以 `GP106 + CC 6.1` 为准。

### 2.2 从整卡到 SM

NVIDIA 给出的整卡配置是 10 个 SM 和 1280 个 CUDA Core，因此每个 SM 有 128 个 CUDA Core。这个结果也与 CUDA Programming Guide 对 Compute Capability 6.1 的定义一致：[每个 SM 有 128 个算术 CUDA Core、32 个特殊函数单元和 4 个 warp scheduler](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#architecture-6-x)。

CUDA Core 不是能够独立执行任意程序的 CPU core。CUDA 线程以 32 个线程组成的 warp 被调度；一个 SM 的 4 个 warp scheduler 向对应执行资源发射指令。因此，“1280 CUDA Core”只描述某类并行算术资源的规模，不能单独预测任意 kernel 的速度。

CC 6.1 SM 的几个重要资源上限如下：

| 每个 SM 的资源 | 数值 |
|---|---:|
| FP32 算术 CUDA Core | 128 |
| FP64 算术 core | 4 |
| 特殊函数单元 | 32 |
| Warp scheduler | 4 |
| 最大常驻 warp | 64 |
| 寄存器文件 | 64K 个 32-bit 寄存器 |
| 统一 L1/Texture Cache | 48 KB |
| Shared Memory | 96 KB |
| 单个 thread block 可用 Shared Memory 上限 | 48 KB |

这些值来自 [CUDA 11.8 Programming Guide 的 CC 6.x 架构说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#architecture-6-x)和 [Pascal Tuning Guide 的 occupancy / shared-memory 说明](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html#sm-occupancy)。寄存器、shared memory、block 大小和每线程寄存器用量会共同限制同一 SM 上能同时驻留的 warp 数。

## 3. 时钟与理论 FP32 吞吐

### 3.1 计算口径

标准参考频率为：

- Base Clock：1506 MHz；
- Boost Clock：1708 MHz。

两项均来自 [NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/)。理论 FP32 峰值假设全部 1280 个 FP32 lane 每周期都完成一次 FMA；按照常见吞吐口径，一次 FMA 包含一次乘法和一次加法，计作 2 FLOP。

基础频率理论值为：

$$ P_{\mathrm{FP32,base}}=1280\times 2\times 1.506\times 10^9=3.85536\times 10^{12}\ \mathrm{FLOP/s}\approx 3.86\ \mathrm{TFLOP/s}. $$

Boost 频率理论值为：

$$ P_{\mathrm{FP32,boost}}=1280\times 2\times 1.708\times 10^9=4.37248\times 10^{12}\ \mathrm{FLOP/s}\approx 4.37\ \mathrm{TFLOP/s}. $$

所以应明确写成：

- **Base 理论 FP32：约 3.86 TFLOP/s**；
- **Boost 理论 FP32：约 4.37 TFLOP/s**。

不能只写一个“约 4.4 TFLOPS”而省略频率假设。CUDA Programming Guide 的原生指令吞吐表也表明，CC 6.1 每个 SM 每周期可产生 128 个 FP32 add、multiply 或 multiply-add 结果；整卡 10 个 SM 与上面的计算一致。[来源：CUDA 11.8 Programming Guide，Table 3](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#arithmetic-instructions)

### 3.2 为什么实际值不会固定等于理论值

上述数字是假设所有 FP32 通道持续执行 FMA 的上限，不是应用性能承诺。NVIDIA 对 GPU Boost 3.0 的说明是根据工作负载动态提高时钟，并涉及温度目标和风扇控制；因此持续频率会受到功耗、温度和工作负载性质影响。[来源：NVIDIA GTX 1060 用户指南，Features](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf)

实际 kernel 还可能因为内存等待、依赖链、分支分歧、低 occupancy、指令混合或 launch 开销而无法每周期让全部 FP32 通道执行 FMA。只做加法或乘法时，每条指令也只有 1 FLOP，不能沿用 FMA 的 2 FLOP 计数。

## 4. 显存、带宽与缓存

### 4.1 6 GB GDDR5 与 192 GB/s

标准配置是 6 GB GDDR5，数据率为 8 Gbit/s，接口宽度为 192 bit。理论带宽按每个引脚的有效数据率计算：

$$ B_{\mathrm{VRAM}}=\frac{8\times 10^9\ \mathrm{bit/s}\times 192\ \mathrm{bit}}{8\ \mathrm{bit/byte}}=192\times 10^9\ \mathrm{byte/s}=192\ \mathrm{GB/s}. $$

结果与 NVIDIA 标称的 192 GB/s 一致。[来源：NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/)

这里的 8 Gbit/s 是 GDDR5 的有效数据率，不应误写成 8 GB/s；192 GB/s 是整条 192-bit 显存接口聚合后的理论峰值。协议开销、访问粒度、读写混合、地址合并程度和 bank/partition 利用率都会使有效带宽低于该值。NVIDIA 的 CUDA Best Practices Guide 也要求把理论带宽与根据实际读写字节数计算的有效带宽分开。[来源：CUDA C++ Best Practices Guide，Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth)

### 4.2 GB 与 GiB

NVIDIA 产品规格使用的名称是“6 GB”。按 NIST 列出的标准前缀定义：

- $1\ \mathrm{GB}=10^9\ \mathrm{byte}$；
- $1\ \mathrm{GiB}=2^{30}\ \mathrm{byte}=1{,}073{,}741{,}824\ \mathrm{byte}$。

[来源：NIST Prefixes for binary multiples](https://physics.nist.gov/cuu/Units/binary.html)

因此，严格按单位换算，$6\ \mathrm{GB}\approx 5.588\ \mathrm{GiB}$，而 $6\ \mathrm{GiB}\approx 6.442\ \mathrm{GB}$。本文保留厂商的“6 GB”写法，不把它无条件改写成“6 GiB”。做容量预算时，应读取 CUDA 或驱动报告的原始字节数；应用实际可分配空间还会小于设备报告总量，因为显示、驱动、CUDA context 和内存分配器都会占用空间。

### 4.3 缓存层级

GP106 整卡有 NVIDIA 标为 1536 KB 的共享 L2 Cache。[来源：NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/)

CC 6.1 的每个 SM 还有 48 KB 统一 L1/Texture Cache 和 96 KB 独立 Shared Memory；L2 则由全部 SM 共享。Pascal 的 L1 与纹理缓存承担合并访问等功能，thread-local memory 和 global memory 的实际缓存行为还取决于 load 类型与编译配置。[来源：CUDA 11.8 Programming Guide，CC 6.x](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#architecture-6-x)；[Pascal Tuning Guide，Unified L1/Texture Cache](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html#l1-cache)

缓存命中和 shared-memory 复用能够减少对 192 GB/s 外部显存的压力，但 48 KB、96 KB 和 1536 KB 这些容量本身不能预测命中率；访问局部性、工作集和 kernel 实现才决定其效果。

## 5. FP32、FP64、FP16、INT8 与现代 AI 数据类型

### 5.1 各算术路径的硬件特征

CUDA Programming Guide 的原生指令吞吐表按“每个 SM 每周期产生的结果数”列出 CC 6.1 的 FP16、FP32、FP64 吞吐分别为 2、128、4。[来源：CUDA 11.8 Programming Guide，Table 3](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#arithmetic-instructions)

| 类型 | CC 6.1 原生结果数/周期/SM | 相对 FP32 吞吐 | 对 GTX 1060 的含义 |
|---|---:|---:|---|
| FP32 | 128 | $1$ | 主要通用浮点路径 |
| FP64 | 4 | $1/32$ | 支持双精度，但吞吐被显著削弱 |
| FP16 | 2 | $1/64$ | 可存储并执行原生半精度指令，但不是高吞吐训练路径 |
| INT8 | `dp4a` / `dp2a` | 专用点积路径 | 适合能够映射到 packed integer dot product 的量化计算 |

以相同频率和 FMA 计数口径推导，FP64 理论峰值约为：

- Base：$3.85536/32=0.12048\ \mathrm{TFLOP/s}$；
- Boost：$4.37248/32=0.13664\ \mathrm{TFLOP/s}$。

FP16 理论算术吞吐则约为：

- Base：$3.85536/64=0.06024\ \mathrm{TFLOP/s}$；
- Boost：$4.37248/64=0.06832\ \mathrm{TFLOP/s}$。

这解释了一个容易误判的现象：FP16 数据只占 FP32 一半空间，可以降低显存容量和带宽压力，但在 GTX 1060 上不会像现代 AI GPU 那样带来更高的 FP16 算术峰值。NVIDIA 的 Pascal Tuning Guide 同样明确指出，消费级 CC 6.1 分支的 FP16 吞吐为 FP32 的 $1/64$。[来源：Pascal Tuning Guide，FP16 Arithmetic Support](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html#fp16)

### 5.2 INT8 是 DP4A，不是 Tensor Core

Pascal CC 6.1 提供 `dp4a` 和 `dp2a`。其中 `dp4a` 在一条指令中计算 4 对 8-bit 整数的点积，并累加到 32-bit 整数；NVIDIA 将这类指令定位于深度学习推理。[来源：Pascal Tuning Guide，INT8 Dot Product](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html#int8)

这是一条 packed SIMD 整数点积路径，不是矩阵乘加单元。只有当软件、数据布局和算子真正映射到 `dp4a`/`dp2a` 时，INT8 才能利用这项能力；“支持 INT8”不能直接等同于现代 Tensor Core GPU 的 INT8 TOPS。

### 5.3 没有 Tensor Core，也没有原生 BF16

NVIDIA 的代际对比表把 GTX 10 系列标为 Pascal，并在 Tensor Core 一栏明确标记为无；Tensor Core 出现在后续架构中。[来源：NVIDIA GeForce 代际对比](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)；[NVIDIA Volta 架构页](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)

CUDA 对 BF16 的硬件要求是 Compute Capability 8.0 或更高，而 GTX 1060 是 6.1，因此它没有原生 CUDA BF16 算术支持。[来源：CUDA Programming Guide，Supported Floating-Point Types](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html#supported-floating-point-types)

软件仍可把 BF16 位模式当作普通数据保存、转换或模拟计算，但这不等于硬件原生 BF16，也不会产生 BF16 Tensor Core 加速。

## 6. PCIe、功耗与显示能力

### 6.1 主机互连

NVIDIA 参考卡使用 PCI Express 3.0 x16。[来源：NVIDIA GTX 1060 用户指南，Hardware Installation](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf)

PCIe 是主机与显卡之间的链路，192 GB/s 则是 GPU 与板载 GDDR5 之间的理论带宽，两者不能混为一谈。频繁的 CPU-GPU 数据往返可能让 PCIe 成为瓶颈；CUDA 文档建议尽量减少 host-device 传输，并在适用时使用 page-locked memory 和异步重叠。[来源：CUDA C++ Best Practices Guide，Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)

### 6.2 功耗与散热边界

NVIDIA 标出的显卡功率为 120 W、最高 GPU 温度为 94 °C，并要求一个 6-pin PCIe 辅助供电接口；官方建议整机电源为 400 W，该建议基于配有 3.2 GHz Intel Core i7 的参考系统。[来源：NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/)

120 W 是显卡级功率规格，不是整机墙上功耗。持续计算是否能保持高时钟还取决于温度、功率限制和散热条件，因此 Boost 理论吞吐不应视为无限时长的稳定吞吐。

### 6.3 显示输出

NVIDIA 参考设计提供 3 个 DisplayPort、1 个 HDMI 和 1 个双链路 DVI；官方规格列出 DisplayPort 1.2 认证且兼容 1.3/1.4、HDMI 2.0b、最多 4 台显示器，以及最高数字分辨率 7680 x 4320 @ 60 Hz。该最高分辨率需要满足官方脚注所列的连接与色彩格式条件。[来源：NVIDIA GTX 1060 官方规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/)；[NVIDIA GeForce 10 系列规格对比](https://www.nvidia.com/zh-tw/geforce/products/10series/compare/)；[NVIDIA GTX 1060 用户指南](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf)

## 7. 图形卡定位与通用计算能力

GTX 1060 的官方定位是高画质 PC 游戏与 VR，而不是 Tesla 数据中心加速器。[来源：NVIDIA GTX 1060 发布页](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/) 其 80 个纹理单元、48 个 ROP、显示引擎和图形 API 支持属于图形吞吐体系；1280 个 CUDA Core、shared memory、cache 和 CUDA 指令集则允许它执行通用并行计算。

从硬件配比看，它明显优先考虑图形与 FP32：

- FP32 吞吐是主要算术能力；
- FP64 只有 FP32 的 $1/32$，不适合高双精度吞吐的科学计算；
- FP16 只有 FP32 的 $1/64$，不具备现代训练卡的混合精度算力优势；
- INT8 有 `dp4a`/`dp2a`，但没有 Tensor Core；
- 没有原生 BF16；
- 只有 6 GB 级显存容量和 PCIe 3.0 主机互连。

因此，它仍可用于 CUDA 学习、较小规模的 FP32 计算、部分图像处理和能够利用 `dp4a` 的量化推理，但不能因为“支持 CUDA/FP16/INT8”就把它等同于现代 AI 加速器。

## 8. 实际硬件瓶颈

### 8.1 显存容量

6 GB 标称容量首先限制能同时驻留的模型参数、梯度、优化器状态、激活、临时 workspace 和框架运行时开销。即使某个算子的计算量不大，只要工作集无法装入显存，也会失败或需要分块、卸载和重计算。

### 8.2 显存带宽

对低算术强度 kernel，192 GB/s GDDR5 带宽通常比 FP32 算术单元更早成为上限。用理论 FP32 峰值除以理论显存带宽，可得到一个仅用于初步判断的 roofline 平衡点：

$$ I_{\mathrm{base}}=\frac{3.85536\ \mathrm{TFLOP/s}}{192\ \mathrm{GB/s}}\approx 20.08\ \mathrm{FLOP/byte},\qquad I_{\mathrm{boost}}=\frac{4.37248\ \mathrm{TFLOP/s}}{192\ \mathrm{GB/s}}\approx 22.77\ \mathrm{FLOP/byte}. $$

若每个从 GDDR5 传输的字节只能支持远低于约 20--23 次 FP32 运算，工作负载更可能受显存带宽限制；若数据能在寄存器、shared memory 或 cache 中大量复用，则更可能接近计算侧限制。这个判断同时使用了两个理论峰值，不能代替 profiler 数据。

### 8.3 并行度与 SM 资源

整卡只有 10 个 SM。小 grid、每个 block 资源消耗过高、寄存器压力、shared-memory 占用或 warp 分歧都可能减少可运行 warp，无法隐藏流水线与内存延迟。NVIDIA 也明确指出，低 occupancy 可能导致指令发射效率差，但 occupancy 足够后继续提高并不保证更快。[来源：NVIDIA Nsight Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)

### 8.4 精度路径与软件映射

FP64、FP16 和 INT8 走不同吞吐路径。FP16 可能节省容量和流量，却不提升该卡的算术峰值；INT8 只有映射到专用点积指令才有收益；依赖 Tensor Core、BF16 或新架构矩阵指令的算法无法在硬件上获得相同路径。

### 8.5 主机传输、功率与温度

频繁 PCIe 传输、同步和短 kernel 启动会降低端到端利用率。长时间高负载下，功率和温度又会影响动态频率。因此“单个 kernel 的理论上限”和“完整应用的持续吞吐”必须分开。

## 9. 标称规格不能预测什么

仅凭 CUDA Core 数、Boost 频率、TFLOP/s 或显存带宽，无法可靠预测：

1. 某个游戏的帧率或帧时间，因为它还取决于着色器、纹理、ROP、CPU、驱动、图形 API、分辨率和画质设置；
2. 某个 CUDA kernel 的时间，因为它还取决于指令混合、数据依赖、occupancy、warp 分歧和访存合并；
3. 矩阵乘的实际 TFLOP/s，因为矩阵形状、布局、库版本、workspace 和精度路径会影响算法选择；
4. 训练或推理能否装入显存，因为权重之外还有激活、梯度、优化器状态、临时 workspace 和运行时开销；
5. 持续时钟和持续功耗，因为 Boost 是动态机制；
6. CPU-GPU 端到端吞吐，因为 PCIe 传输、同步、数据准备和 kernel launch 不包含在芯片 FP32 峰值中；
7. 有效显存带宽，因为 192 GB/s 只是接口理论值，访问模式、缓存命中和实际读写量决定有效值；
8. 跨代性能，因为不同架构中“一个 CUDA Core”的调度、数据通路、缓存和专用单元并不等价。

## 10. 使用这些规格时的正确口径

- 写 FP32 峰值时同时标明 **Base 约 3.86 TFLOP/s** 与 **Boost 约 4.37 TFLOP/s**。
- 写显存时使用厂商原称 **6 GB GDDR5**；需要精确容量预算时改查原始 byte 数，不把 GB 与 GiB 混写。
- 写带宽时标明 **192 GB/s 理论显存带宽**，不要把它当作 PCIe 带宽或应用有效带宽。
- 写 FP16 时同时注明 **可存储/可执行** 与 **算术吞吐仅为 FP32 的 $1/64$**。
- 写 INT8 时注明它是 `dp4a`/`dp2a` packed dot product，不是 Tensor Core。
- 写 AI 能力时明确 **无 Tensor Core、无原生 BF16**。
- 用 benchmark 或 profiler 回答实际性能问题；规格表只给出容量、功能和理论上界。

## 11. 一手资料索引

1. [NVIDIA：GeForce GTX 1060 发布页与详细芯片规格](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/)
2. [NVIDIA：GeForce GTX 1060 官方完整规格页](https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1060/)
3. [NVIDIA：GeForce GTX 1060 6 GB Founders Edition User Guide](https://www.nvidia.com/content/geforce-gtx/GEFORCE_GTX_1060_USER_GUIDE_v02.pdf)
4. [NVIDIA：Legacy CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus/legacy)
5. [NVIDIA CUDA 11.3：Pascal Tuning Guide](https://docs.nvidia.com/cuda/archive/11.3.0/pascal-tuning-guide/index.html)
6. [NVIDIA CUDA 11.8：CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html)
7. [NVIDIA：Pascal GP100 Architecture Whitepaper v1.2](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)
8. [NVIDIA：GeForce 架构代际对比](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)
9. [NVIDIA CUDA 13.1：Floating-Point Computation 与 BF16 要求](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html)
10. [NVIDIA CUDA：CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
11. [NVIDIA Nsight：Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)
12. [NIST：Prefixes for binary multiples](https://physics.nist.gov/cuu/Units/binary.html)
