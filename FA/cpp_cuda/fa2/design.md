# FP32 CUDA Flash-Attention Baseline

## 1. Purpose

This directory establishes a correctness-first CUDA baseline that can be
optimized toward FlashAttention-2 in measured steps.

The current implementation provides the essential memory algorithm:

- Exact scaled dot-product attention.
- Online softmax.
- No materialized `(S, S)` scores or probabilities.
- Causal and non-causal forward.

It intentionally does not begin with production FA2 optimizations. Each later
change can therefore be benchmarked and profiled independently.

## 2. Supported Interface

- Q/K/V/O shape: `(S, d)`.
- FP32 only.
- One self-attention head.
- Contiguous CUDA tensors.
- `1 <= d <= 128`.
- Forward only.

There is no backward, dropout, arbitrary mask, GQA, KV cache, FP16/BF16 or
Tensor Core path.

[binding.cpp](./binding.cpp) exposes the C++ entry point.
[fa2_forward.cu](./fa2_forward.cu) contains the baseline kernel.
[build_extension.py](./build_extension.py) builds it without Ninja.
The Python wrapper is
[cuda_fa2_forward.py](../../src/cuda_fa2_forward.py).

## 3. Baseline Work Partition

The launch uses one 128-thread CTA per query row:

```text
grid.x = S
block.x = 128
```

For each visible key:

1. Threads compute one element of `Q_i * K_j`.
2. Shared-memory tree reduction produces the dot product.
3. Thread 0 updates the online-softmax state.
4. Threads update one output dimension with `V_j`.

K/V are read directly from global memory for every query row. There is no
cross-row reuse, vectorized loading or shared-memory K/V tiling yet.

## 4. Online Softmax

For each query row, the kernel maintains running maximum `m`, normalizer `l`
and unnormalized output `o`. For a new score `x`:

```text
m_new = max(m, x)
alpha = exp(m - m_new)
beta  = exp(x - m_new)
l     = alpha * l + beta
o     = alpha * o + beta * V_j
m     = m_new
```

After all visible keys, it writes `o / l`. Arithmetic complexity remains
`Theta(S^2 d)`, but global auxiliary storage is `Theta(Sd)` rather than
`Theta(S^2)`.

## 5. Planned Optimization Ladder

The baseline leaves the following changes measurable:

1. Replace the shared-memory tree reduction with warp shuffle reduction.
2. Process multiple query rows per CTA.
3. Load K/V tiles once into shared memory and reuse them across query rows.
4. Vectorize Q/K/V global loads where alignment permits.
5. Double-buffer K/V tiles to overlap loads and computation.
6. Tune Q/K tile sizes using register, shared-memory and occupancy evidence.
7. Add specialized kernels for common head dimensions.
8. Add lower-precision Tensor Core paths on supported GPUs.

Each step should record numerical error, CUDA Event latency, kernel duration,
launch gaps, memory traffic, occupancy and peak allocation before proceeding.

## 6. Build

Ninja is not required. From the `FA` directory:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=6.1 \
/home/dengxiao/miniconda3/envs/nanovllm/bin/python -E -s -B \
  cpp_cuda/fa2/build_extension.py build_ext --inplace
```

The build writes the ignored extension binary beside the Python package and
uses the ordinary setuptools build directory for intermediates. Build before
running tests, benchmark or profiling.

The CUDA source uses `-lineinfo`, so Nsight Systems can identify
`fa2_forward_fp32_kernel`.
