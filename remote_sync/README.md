# Profile 源码同步工具

`sync_to_cuda.sh` 只负责把 Assignment 2 profile 所需源码从开发机 B 单向同步到 `cuda-via-a`，不安装依赖、不创建 Python 环境，也不运行 benchmark 或 profile。

## 同步范围

```text
assignment1-basics/cs336_basics/**
assignment2-systems/cs336_systems/**
assignment2-systems/benchmark.py
assignment2-systems/remote_profile/run_small_benchmark.sh
```

默认目标：

```text
/home/dengxiao/work/cs336-profile/assignments/
```

目标是专用、可重建的代码目录。脚本使用 `--delete-excluded` 保证其中不会积累不在白名单内的文件，因此不要把日志、profile 或人工修改放进该目录。

## 使用

在开发机 B 的 Assignment 2 根目录运行。

只预览，不修改 C：

```bash
./remote_sync/sync_to_cuda.sh
```

预览、人工确认并同步：

```bash
./remote_sync/sync_to_cuda.sh --apply
```

非交互同步：

```bash
./remote_sync/sync_to_cuda.sh --apply --yes
```

覆盖 SSH 别名或目标根目录：

```bash
./remote_sync/sync_to_cuda.sh \
  --remote cuda-via-a \
  --remote-root /home/dengxiao/work/cs336-profile/assignments
```

目标根目录必须以 `/work/cs336-profile/assignments` 结尾，避免错误的 `--delete-excluded` 目标。

## 安全与幂等性

- 默认模式是 dry-run。
- `--apply` 在写入前仍会执行一次远端 dry-run。
- 交互模式必须明确输入 `y` 才执行。
- 使用本地文件锁阻止两个同步进程并发运行。
- 使用内容校验，不依赖两台机器的文件时间是否一致。
- 使用延迟更新和延迟删除，减少中途失败造成的半更新状态。
- 正式同步后再次 dry-run；无变化才报告 `idempotency_check=passed`。
- 不执行任何 C 到 B 的反向同步。

## 前置条件

1. A 已建立到 B 的反向 SSH 隧道。
2. B 的 `cuda-via-a` SSH 别名可连接 C。
3. B 和 C 都已安装 `rsync`。
4. 已在 C 的 `cs336` tmux session 中创建两个目标项目目录。
