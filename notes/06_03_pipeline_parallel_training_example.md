# 最小 Pipeline Parallel 训练示例

## 1. 这个脚本展示什么

[`train_pipeline_parallel.py`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py) 只展示一条完整的 PP 训练路径：

1. 启动两个分布式进程；
2. 每个 rank 只构造自己负责的 Transformer stage；
3. 将一个 mini-batch 切成 4 个 microbatch；
4. Forward 时从前一 stage 接收 activation，并发送给后一 stage；
5. Backward 时沿反方向发送 activation gradient；
6. 每个 stage 使用累积后的本地参数梯度执行一次 `optimizer.step()`。

脚本没有 benchmark、内存统计或 baseline 参数 diff。

## 2. 直接运行

在 assignment 目录执行：

```bash
uv run python scripts/train_pipeline_parallel.py
```

若 PyTorch 支持 NCCL 且至少有两张可见 CUDA GPU，脚本自动使用 NCCL；否则使用 CPU/Gloo。为避免 CPU 进程过度占用资源，每个 worker 固定使用一个 PyTorch CPU thread。

本机 CPU/Gloo 实际输出为：

```text
rank=1 layers=[2,4) parameter_count=98624
rank=0 layers=[0,2) parameter_count=98560
step=0 loss=5.764530
step=1 loss=5.822822
step=2 loss=5.773110
```

两个 rank 的日志顺序可能变化，因为它们是并发进程。

## 3. 模型如何分到两个 Rank

脚本的固定配置位于 [`train_pipeline_parallel.py:L16-L22`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L16-L22)：

```text
world size:       2
Transformer 层数: 4
mini-batch size:  8
microbatch 数:    4
```

因此每个 microbatch 包含 2 个样本，模型按连续层均分：

```text
rank 0 / stage 0:
  token embedding
  Transformer layers [0, 2)

rank 1 / stage 1:
  Transformer layers [2, 4)
  final RMSNorm
  lm_head
```

[`TransformerPipelineStage.from_dimensions()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/pipeline_parallel.py#L132-L171) 只创建本 rank 所属的模块，不会先在每个 rank 上构造完整模型。

对应调用位于 [`train_pipeline_parallel.py:L41-L56`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L41-L56)：

```python
stage = TransformerPipelineStage.from_dimensions(
    stage_index=rank,
    num_stages=WORLD_SIZE,
    ...
)
pipeline = PipelineParallel(stage)
optimizer = AdamW(stage.parameters(), lr=1e-3)
```

每个 optimizer 只接收本地 `stage.parameters()`，因此 rank 0 不保存或更新 layers 2、3，rank 1 也不保存或更新 layers 0、1。

## 4. 分布式进程如何启动

[`main()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L85-L94) 完成两件事：

1. 根据可见 GPU 数选择 NCCL 或 Gloo；
2. 使用 `mp.spawn()` 启动两个 `train_worker` 进程。

每个 worker 在 [`train_pipeline_parallel.py:L29-L36`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L29-L36) 中加入同一个 process group：

```python
dist.init_process_group(
    backend,
    store=store,
    rank=rank,
    world_size=WORLD_SIZE,
)
```

`rank` 同时也是 stage index，所以通信 peer 很直接：

```text
rank 0 的 next rank = 1
rank 1 的 previous rank = 0
```

## 5. 一步训练只包含三个调用

训练循环的核心位于 [`train_pipeline_parallel.py:L69-L76`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L69-L76)：

```python
optimizer.zero_grad(set_to_none=True)
loss = pipeline.forward_backward(
    input_ids,
    targets,
    loss_fn=language_model_loss,
    num_microbatches=NUM_MICROBATCHES,
)
optimizer.step()
```

这三步的职责分别是：

1. `zero_grad()`：清除上一个 mini-batch 的本地参数梯度；
2. `forward_backward()`：执行所有 microbatch 的流水线 forward/backward，并累积本地参数梯度；
3. `optimizer.step()`：每个 rank 只更新自己 stage 的参数。

`optimizer.step()` 必须等所有 microbatch backward 完成后执行。这样同一个 mini-batch 的所有 microbatch 都使用同一版本参数。

## 6. `forward_backward()` 内部发生什么

[`PipelineParallel.forward_backward()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/pipeline_parallel.py#L210-L238) 首先把 batch size 8 切成 4 个大小为 2 的 microbatch。

### 6.1 Forward 填充

它依次执行：

```text
F0 -> F1 -> F2 -> F3
```

对每个 microbatch：

```text
rank 0:
  token IDs
  -> embedding + layers 0,1
  -> send hidden activation to rank 1

rank 1:
  recv hidden activation from rank 0
  -> layers 2,3 + final norm + lm_head
  -> compute scaled loss
```

具体的发送和接收位于：

- [`_forward_microbatch()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/pipeline_parallel.py#L253-L279)
- [`_receive_input_activation()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/pipeline_parallel.py#L281-L290)

Forward 结束后，每个 stage 保存 4 个局部 autograd graph，供之后 backward 使用。

### 6.2 Backward 排空

随后按逆序执行：

```text
B3 -> B2 -> B1 -> B0
```

对每个 microbatch：

```text
rank 1:
  loss.backward()
  -> obtain input_activation.grad
  -> send activation gradient to rank 0

rank 0:
  recv activation gradient from rank 1
  -> local_output.backward(received_gradient)
  -> accumulate gradients for embedding + layers 0,1
```

这部分对应 [`_backward_microbatch()`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/cs336_systems/pipeline_parallel.py#L292-L306)。

## 7. 为什么只有 Rank 1 打印 Loss

只有最后一个 stage 拥有 `lm_head`，能够生成 logits 并调用 `language_model_loss()`。因此：

```text
rank 0: forward_backward() 返回 None
rank 1: forward_backward() 返回完整 mini-batch 的平均 loss
```

每个 microbatch loss 在 backward 前已经除以 4，最后将 4 个 scaled loss 相加：

$$ L=\frac{L_0}{4}+\frac{L_1}{4}+\frac{L_2}{4}+\frac{L_3}{4}. $$

所以 [`train_pipeline_parallel.py:L78-L80`](file:///home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment2-systems/scripts/train_pipeline_parallel.py#L78-L80) 只在 `loss is not None` 时打印。

## 8. PP 的核心本质

去掉进程初始化和随机数据后，这个例子只剩下四条原则：

1. **模型状态按层切分**：每个 rank 只持有一个 stage。
2. **Forward 发送 activation**：后一 stage 不需要前一 stage 的参数。
3. **Backward 发送 activation gradient**：前一 stage 用它继续本地反向传播。
4. **参数更新保持本地**：所有 microbatch 梯度累积完成后，每个 rank 更新自己的参数。

这就是当前实现所展示的 Pipeline Parallelism。
