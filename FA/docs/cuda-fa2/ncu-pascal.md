# 1. GTX 1060 的 Nsight Compute 隔离安装

验证日期：2026-10-11。设备：GTX 1060（GP106、CC 6.1）。

**Nsight Compute 2019.5.0 已独立安装，并成功通过 CUDA Driver API 采集现有 FA kernel。**
当前驱动 `570.211.01`、CUDA Toolkit 12.8、PyTorch `2.6.0+cu124` 保持原样，
默认 `ncu` 仍指向 2025.1.1。
旧版直接包装当前 PyTorch 入口会在 CUDA 初始化阶段返回 error 36；
可用方式是通过独立 Driver API 入口加载现有扩展中的 `sm_61` cubin。

## 1.1 安装位置与官方来源

| 项目 | 位置 |
|---|---|
| 独立解包目录 | `/home/dengxiao/.local/opt/nsight-compute-2019.5.0` |
| CLI 包装命令 | `/home/dengxiao/.local/bin/ncu-pascal` |
| GUI 包装命令 | `/home/dengxiao/.local/bin/ncu-pascal-ui` |
| 原版 CLI | `…/opt/nvidia/nsight-compute/2019.5.0/nv-nsight-cu-cli` |

安装仅使用 `dpkg-deb -x` 解包官方包到用户目录，没有执行系统包安装或维护脚本，
也没有修改 shell 配置、全局 `PATH`、`LD_LIBRARY_PATH`、CUDA 链接或驱动设置。
两个包装命令只清理自身进程继承的 `LD_LIBRARY_PATH`，避免开发工具附带库影响旧版程序加载。

官方安装包：
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/nsight-compute-2019.5.0_2019.5.0.14-1_amd64.deb

包的 SHA-256 已与 NVIDIA 仓库元数据核对：

```text
2b934b30e88de4861773bdf274f994099ccf06c0119627e6db2f9909d70ca763
```

[官方包元数据](assets/ncu-pascal/package-metadata.txt)记录了版本 `2019.5.0.14-1`。
安装包约 204 MiB，解包后约 474 MiB；没有创建额外 Python 环境。
官方旧版 [GPU 支持表](https://docs.nvidia.com/nsight-compute/2019.5.1/ReleaseNotes/index.html#gpu-support)
列出 Pascal GP10x 为支持，而 GP100 不支持。

## 1.2 实际兼容性结果

| 验证 | 结果 |
|---|---|
| `ncu-pascal --version` | 2019.5.0，正常启动 |
| `ncu-pascal --query-metrics` | 正确识别 Device GP106 并列出指标 |
| 包装现有 `scripts.profile_attention` | PyTorch CUDA 初始化报 error 36，未采集 |
| Driver API 加载同一个 FA cubin | 成功运行，校验通过 |
| Driver API + `--set detailed` | 成功采集 31 个 replay passes，生成报告 |
| CLI 重新导入生成的报告 | 成功 |
| 原 PyTorch CUDA 测试 | 4 项全部通过 |
| CUDA 路径、当前 ncu/nvcc、驱动库和 FA 扩展 | realpath / SHA-256 均与安装前一致 |

因此，旧版可以在这套软件栈旁独立使用，但不能直接代替现代 profiler 包装当前 PyTorch 程序。
旧版与 CUDA 12 程序组合出现 API 不兼容，也有 NVIDIA 论坛中的相同案例；具体能否运行仍取决于采集入口。[cite:1]

[安装前文件记录](assets/ncu-pascal/stack-before.json)用于核对软件栈；
当前默认 `ncu` 仍为 2025.1.1，没有切换版本。
GUI 所需共享库能通过安装目录中的附带库解析，但本次未在图形会话中交互操作 GUI。

## 1.3 可直接使用的采集命令

以下命令从 `FA` 目录执行，先用小输入学习各个报告页面：

```bash
sudo /home/dengxiao/.local/bin/ncu-pascal \
  --profile-from-start off \
  --kernel-regex fa2_forward_fp32_kernel \
  --launch-count 1 \
  --clock-control none \
  --set detailed \
  --page details \
  --export /tmp/fa2-ncu-pascal \
  /home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  -m scripts.profile_attention_driver \
  --seq-len 64 --head-dim 64 --warmup 1 --no-causal
```

驱动仍设置为仅管理员可读性能计数器，所以采集使用 sudo。
`--clock-control none` 避免 profiler 主动锁定 GPU 频率。
`--page details` 提供按指标逐行显示的文本表格；
报告文件扩展名是旧版的 `.nsight-cuprof-report`。

已有成功采集结果：

- [Nsight Compute 报告](assets/ncu-pascal/fa2-driver-s64-d64.nsight-cuprof-report)
- [可读文本结果](assets/ncu-pascal/fa2-driver-s64-d64.txt)

在具备图形显示的 Linux 会话中，可用独立 GUI 打开：

```bash
/home/dengxiao/.local/bin/ncu-pascal-ui \
  docs/cuda-fa2/assets/ncu-pascal/fa2-driver-s64-d64.nsight-cuprof-report
```

报告包含 GPU Speed Of Light、Compute/Memory Workload Analysis、
Scheduler Statistics、Warp State Statistics、Instruction Statistics、Launch Statistics 和 Occupancy。
旧版 Pascal 的指标名称与新版架构不同，查询指标时应使用该旧版 CLI 的 `--query-metrics`。

## 1.4 Driver API 入口及数据边界

[独立采集入口](file:///home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA/scripts/profile_attention_driver.py)
只使用 Python 标准库，不导入 PyTorch，也不加载它的 CUDA Runtime。

1. [从当前扩展提取 cubin](file:///home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA/scripts/profile_attention_driver.py#L63-L81)：
   每次运行都提取 `src/cuda_fa2_extension*.so` 中的 `sm_61` 二进制，并发现真实 kernel 符号。
2. [通过 Driver API 启动](file:///home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA/scripts/profile_attention_driver.py#L103-L145)：
   建立独立 context、分配 Q/K/V/O、执行 warmup，再捕获一次 forward。
3. [在 capture 外验证结果](file:///home/dengxiao/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/FA/scripts/profile_attention_driver.py#L84-L100)：
   使用 `Q=K=0`、确定性的 V，因此输出应为 V 的均值；causal 模式为前缀均值。

已验证 non-causal `(64,64)` 和 causal `(7,3)`，最大绝对误差分别为 0 与约 `6.39e-9`。
这些是工具兼容性和学习用例，**输入不同于原随机输入 baseline**；
其延迟、occupancy 或 stall 数值不能直接替代 [V0 性能分析](measurements/v0/perf-baseline.md) 的性能数据。

入口保留 baseline 的 `128 threads/CTA` 和既有参数 ABI。
修改 kernel 后应先重新构建扩展；若修改线程组织或参数接口，还应同步更新这个启动入口。
该脚本不负责构建，也没有使用另一份重新编译的 kernel。

输入准备与均值校验复杂度为 `O(Sd)`，校验额外空间为 `O(d)`；
GPU kernel 仍是原实现的 `O(S²d)`。CPU 准备与校验为单进程执行，
GPU 并行方式由当前 cubin 决定。多指标 replay 的时间不能作为正式 benchmark 延迟。

## 1.5 参考

<a id="cite-1"></a>[cite:1] NVIDIA Developer Forums：Nsight Compute 2019.5 包装 CUDA 12 程序时出现 API 不兼容的案例。
https://forums.developer.nvidia.com/t/bandwidthtest-example-throws-cudaerrorcallrequiresnewerdriver-error-when-launched-via-nv-nsight-cu-cli/278698

安装包、GPU 支持范围和实测成功记录分别链接于上文；可用性结论以本机实际采集为准。
