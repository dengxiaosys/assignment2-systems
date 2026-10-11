# V2 FAQ

## 1. V2 的优化与 V1 的复用

V2 每 block 的 query 从4行增至8行，每轮[协作加载](snapshot/fa2_forward.cu#L67-L78)32行 K/V 到 shared，供8行复用。

V1 已能通过[缓存复用](../v1/perf_v1.md#memory-throughput) K/V，但各 query warp 仍独立发起 global load；V2 显式共享 tile，减少重复请求，不等于显存流量减少8倍。

另外，32个 lane 并行计算32个 score及指数；m、ℓ、z 改为[每 tile 更新](snapshot/fa2_forward.cu#L81-L122)，减少递推、旧状态缩放与指数计算；[K 转置加 padding](snapshot/fa2_forward.cu#L43-L45)避免 bank 冲突。代价是每 tile 两次屏障和资源占用增加。每 warp 独占一行输出、最终归一化沿用 V1。
