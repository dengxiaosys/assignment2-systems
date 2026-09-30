# CUDA FlashAttention correctness and benchmark harness

This directory contains a standalone CUDA implementation for validating the
FlashAttention algorithm on Pascal GPUs such as the GTX 1060. It does not depend
on the repository's PyTorch CUDA build, so it can be compiled with a separate
CUDA 11.8 toolchain targeting compute capability 6.1.

## Scope

- Single attention head with contiguous row-major FP32 tensors.
- Query shape `(Nq, d)` and key/value shape `(Nk, d)`.
- Causal and non-causal forward/backward.
- Tiled implementation supports `head_dim <= 128`.
- No dropout, batching, MQA/GQA, variable-length packing, or Tensor Core path.
- Causal masking uses `key_index <= query_index`.

The narrow interface is intentional. Batch and head dimensions are independent
outer dimensions and can be added by an adapter without changing the kernels'
single-head algorithm.

## Module layout

| Module | Responsibility |
|---|---|
| `include/fa/shape.h` | Shape invariants and element counts |
| `include/fa/attention.h` | CUDA forward/backward interface and workspace contract |
| `include/fa/reference.h` | CPU correctness-reference interface |
| `src/reference.cpp` | Double-accumulation CPU forward/backward |
| `src/naive.cu` | CUDA baseline that materializes full `S`, `P`, `dP`, and `dS` |
| `src/tiled_forward.cu` | Online-softmax forward without quadratic HBM workspace |
| `src/tiled_backward.cu` | Query-owner and key-owner backward without quadratic HBM workspace |
| `app/benchmark.cu` | Correctness comparison and CUDA-Event benchmark |
| `tests/reference_test.cpp` | CPU finite-difference test |

Both CUDA implementations use the same public interface. The caller allocates
workspace once and passes it into each launch, so timed iterations do not include
`cudaMalloc` or `cudaFree`.

## Build for GTX 1060

Use a CUDA toolkit that still supports Pascal. CUDA 11.8 is the conservative
choice; CUDA 13 does not support `sm_61`.

```bash
cmake -S cuda_flash_attention -B cuda_flash_attention/build \
  -DFA_ENABLE_CUDA=ON \
  -DFA_CUDA_ARCHITECTURE=61
cmake --build cuda_flash_attention/build -j
```

The generated CUDA code uses ordinary FP32 CUDA cores, warp shuffles, and at
most 36 KiB of dynamic shared memory per block. It does not use BF16, Tensor
Cores, `cp.async`, cooperative groups, or architecture-specific asynchronous
pipelines.

## CPU-only reference validation

The CPU reference has no CUDA dependency and can be checked before moving the
source tree to the GPU machine:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -Icuda_flash_attention/include \
  cuda_flash_attention/src/reference.cpp \
  cuda_flash_attention/tests/reference_test.cpp \
  -o cuda_flash_attention/build-host/reference_test

cuda_flash_attention/build-host/reference_test
```

The test compares analytical `dQ`, `dK`, and `dV` against central finite
differences for causal and non-causal attention.

## Correctness and performance

Run both implementations:

```bash
cuda_flash_attention/build/fa_benchmark \
  --nq 512 \
  --nk 512 \
  --head-dim 64 \
  --causal \
  --impl both \
  --warmup 10 \
  --iterations 100
```

Run one implementation:

```bash
cuda_flash_attention/build/fa_benchmark --impl naive
cuda_flash_attention/build/fa_benchmark --impl tiled
```

The executable:

1. generates deterministic input tensors;
2. computes CPU forward/backward references;
3. checks `O`, `L`, `dQ`, `dK`, and `dV`;
4. measures forward and backward with CUDA Events;
5. reports implementation workspace bytes.

Use `--no-verify` for large performance-only cases to avoid the CPU reference
cost.

## Memory model

For FP32:

| Implementation | Forward workspace | Backward workspace |
|---|---:|---:|
| naive | `2 * Nq * Nk * sizeof(float)` | `(4 * Nq * Nk + Nq) * sizeof(float)` |
| tiled | `0` | `Nq * sizeof(float)` for `D = rowsum(O * dO)` |

The tiled kernels reconstruct only the active score/probability/gradient tile.
They never allocate full `S`, `P`, `dP`, or `dS` device buffers.

## Complexity and parallel ownership

Both implementations perform `Theta(Nq * Nk * d)` arithmetic in forward and
backward. The tiled backward has a larger constant because its query-owner and
key-owner passes independently reconstruct local scores and probabilities.

| Path | Output owner | Reduction direction | Global workspace |
|---|---|---|---:|
| naive forward | one thread per output element | all keys | `Theta(Nq * Nk)` |
| naive backward | one thread per gradient element | keys for `dQ`, queries for `dK/dV` | `Theta(Nq * Nk)` |
| tiled forward | one warp per query row | streams 32-key tiles | `0` |
| tiled `dQ` | one warp per query row | all key tiles | `Theta(Nq)` correction buffer |
| tiled `dK/dV` | one warp per key row | all query tiles | same correction buffer |

The tiled output ownership makes every final gradient write race-free without
atomics. At `head_dim=128`, forward uses 34 KiB and backward uses 36 KiB of
dynamic shared memory per block, within the GTX 1060's 48 KiB limit.

## Interpreting results

This is a readable Pascal-oriented implementation, not a replacement for
FlashAttention's production CUDA kernels. Compare naive and tiled results only
on the same GPU, toolkit, shape, dtype, and build flags. Do not extrapolate GTX
1060 timings to A100, H100, B200, Tensor Core paths, or official FlashAttention
implementations.
