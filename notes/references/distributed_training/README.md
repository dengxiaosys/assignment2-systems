# Distributed Training 原始论文

本目录保存与 CS336 Systems 分布式训练章节直接相关的原始论文 PDF，以及便于仓库内搜索的自动文本提取稿。

## 论文索引

| 主题 | 论文 | 原始 PDF | 文本提取稿 | 课程关联 |
|---|---|---|---|---|
| ZeRO | Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | [PDF](./zero_memory_optimizations_toward_training_trillion_parameter_models.pdf) | [Markdown](./zero_memory_optimizations_toward_training_trillion_parameter_models_extracted.md) | optimizer state、gradient、parameter sharding |
| Megatron-LM | Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* | [PDF](./megatron_lm_training_multi_billion_parameter_language_models.pdf) | [Markdown](./megatron_lm_training_multi_billion_parameter_language_models_extracted.md) | tensor parallelism、Transformer MLP/attention 切分 |
| GPipe | Huang et al., *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism* | [PDF](./gpipe_easy_scaling_with_micro_batch_pipeline_parallelism.pdf) | [Markdown](./gpipe_easy_scaling_with_micro_batch_pipeline_parallelism_extracted.md) | pipeline parallelism、micro-batch、pipeline bubble、rematerialization |

## 来源与版本

| 论文 | arXiv 版本 | Canonical URL | PDF 页数 |
|---|---|---|---:|
| ZeRO | [1910.02054v3](https://arxiv.org/abs/1910.02054v3) | `https://arxiv.org/pdf/1910.02054` | 24 |
| Megatron-LM | [1909.08053v4](https://arxiv.org/abs/1909.08053v4) | `https://arxiv.org/pdf/1909.08053` | 15 |
| GPipe | [1811.06965v5](https://arxiv.org/abs/1811.06965v5) | `https://arxiv.org/pdf/1811.06965` | 11 |

这些文件下载于 2026-10-02。GPipe 的 arXiv v5 PDF 标题为 *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*。

## 文件校验

```text
697f257e85667829053707594800447f3a75482226ddb6eb24374905c5af6d44  gpipe_easy_scaling_with_micro_batch_pipeline_parallelism.pdf
fa89e54e23c0cea5cbd9d24720db7647c61af299afdb9a170c7119160e6d207a  megatron_lm_training_multi_billion_parameter_language_models.pdf
c531734c7d9ee647fda9b4f0e1a6abd6d58ad34e862bc945451b09032be3167e  zero_memory_optimizations_toward_training_trillion_parameter_models.pdf
```

## 提取稿说明

`*_extracted.md` 由 PDF 自动提取并按页分节，用于全文搜索和定位原文。自动提取无法可靠保留复杂公式、图、表和多栏排版；引用或推导时应回到同名 PDF 核验。
