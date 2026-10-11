#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int KEYS_PER_TILE = WARP_SIZE;
constexpr int KEY_STRIDE = KEYS_PER_TILE + 1;
constexpr int ACCUMULATORS = 4;
constexpr int MAX_HEAD_DIM = 128;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
static_assert(WARP_SIZE % ACCUMULATORS == 0);

__device__ float warp_max(float value) {
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        value = fmaxf(value, __shfl_xor_sync(FULL_WARP_MASK, value, stride));
    }
    return value;
}

__device__ float warp_sum(float value) {
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        value += __shfl_xor_sync(FULL_WARP_MASK, value, stride);
    }
    return value;
}

template <int COUNT>
__device__ __forceinline__ float sum_partials(float (&partial)[COUNT]) {
    static_assert(COUNT > 0 && (COUNT & (COUNT - 1)) == 0);
#pragma unroll
    for (int stride = COUNT / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int ai = 0; ai < stride; ++ai) {
            partial[ai] += partial[ai + stride];
        }
    }
    return partial[0];
}

template <int DIMS_PER_LANE>
__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                        const float *__restrict__ v, float *const __restrict__ o,
                                        const int seq_len, const int head_dim, const float scale,
                                        const bool is_causal) {
    constexpr int TILE_HEAD_DIM = DIMS_PER_LANE * WARP_SIZE;
    __shared__ float k_tile[TILE_HEAD_DIM][KEY_STRIDE];
    __shared__ float v_tile[KEYS_PER_TILE][TILE_HEAD_DIM];

    const int warpi = threadIdx.x / WARP_SIZE;
    const int lanei = threadIdx.x % WARP_SIZE;
    const int query_base = blockIdx.x * WARPS_PER_BLOCK;
    const int qi = query_base + warpi;
    const bool valid_query = qi < seq_len;

    // Every warp, including query-tail warps, participates in both CTA barriers.
    const int block_key_limit = is_causal ? min(seq_len, query_base + WARPS_PER_BLOCK) : seq_len;
    const int key_limit = valid_query ? (is_causal ? qi + 1 : seq_len) : 0;

    float q_dim[DIMS_PER_LANE];
    float z[DIMS_PER_LANE];
#pragma unroll
    for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
        const int dimi = ri * WARP_SIZE + lanei;
        q_dim[ri] = valid_query && dimi < head_dim ? q[qi * head_dim + dimi] : 0.0f;
        z[ri] = 0.0f;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;

    for (int tilei = 0; tilei < block_key_limit; tilei += KEYS_PER_TILE) {
        // Coalesced global reads; padding removes transpose-store bank conflicts.
        for (int elementi = threadIdx.x; elementi < KEYS_PER_TILE * TILE_HEAD_DIM;
             elementi += THREADS_PER_BLOCK) {
            const int keyi = elementi / TILE_HEAD_DIM;
            const int dimi = elementi % TILE_HEAD_DIM;
            const int ki = tilei + keyi;
            const bool valid_element = ki < block_key_limit && dimi < head_dim;
            k_tile[dimi][keyi] = valid_element ? k[ki * head_dim + dimi] : 0.0f;
            v_tile[keyi][dimi] = valid_element ? v[ki * head_dim + dimi] : 0.0f;
        }
        __syncthreads(); // Publish this tile to all query warps.

        // The predicate is warp-uniform. Inactive warps still reach the reuse barrier.
        if (valid_query && tilei < key_limit) {
            const int ki = tilei + lanei;
            const bool valid_key = ki < key_limit;
            float qk_partial[ACCUMULATORS] = {};
#pragma unroll
            for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
#pragma unroll
                for (int dim_basei = 0; dim_basei < WARP_SIZE; dim_basei += ACCUMULATORS) {
                    float q_values[ACCUMULATORS];
                    float k_values[ACCUMULATORS];
#pragma unroll
                    for (int ai = 0; ai < ACCUMULATORS; ++ai) {
                        const int dim_lanei = dim_basei + ai;
                        const int dimi = ri * WARP_SIZE + dim_lanei;
                        q_values[ai] = __shfl_sync(FULL_WARP_MASK, q_dim[ri], dim_lanei);
                        k_values[ai] = k_tile[dimi][lanei];
                    }
#pragma unroll
                    for (int ai = 0; ai < ACCUMULATORS; ++ai) {
                        qk_partial[ai] += q_values[ai] * k_values[ai];
                    }
                }
            }
            const float dot = sum_partials(qk_partial);
            const float s = valid_key ? dot * scale : -CUDART_INF_F;

            const float m_new = fmaxf(m, warp_max(s));
            float alpha = 0.0f;
            if (lanei == 0) {
                alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
            }
            alpha = __shfl_sync(FULL_WARP_MASK, alpha, 0);
            const float p_tilde = valid_key ? expf(s - m_new) : 0.0f;
            const float l_tile = warp_sum(p_tilde);
            l = alpha * l + l_tile;
            m = m_new;

            // Tile contributions are independent of the persistent output state.
            float pv_partial[DIMS_PER_LANE][ACCUMULATORS] = {};
#pragma unroll
            for (int key_basei = 0; key_basei < KEYS_PER_TILE; key_basei += ACCUMULATORS) {
                float p_values[ACCUMULATORS];
                float v_values[DIMS_PER_LANE][ACCUMULATORS];
#pragma unroll
                for (int ai = 0; ai < ACCUMULATORS; ++ai) {
                    const int keyi = key_basei + ai;
                    p_values[ai] = __shfl_sync(FULL_WARP_MASK, p_tilde, keyi);
#pragma unroll
                    for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
                        const int dimi = ri * WARP_SIZE + lanei;
                        v_values[ri][ai] = v_tile[keyi][dimi];
                    }
                }
#pragma unroll
                for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
#pragma unroll
                    for (int ai = 0; ai < ACCUMULATORS; ++ai) {
                        pv_partial[ri][ai] += p_values[ai] * v_values[ri][ai];
                    }
                }
            }

#pragma unroll
            for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
                z[ri] = alpha * z[ri] + sum_partials(pv_partial[ri]);
            }
        }
        __syncthreads(); // Finish all readers before overwriting shared storage.
    }

    if (valid_query) {
#pragma unroll
        for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
            const int dimi = ri * WARP_SIZE + lanei;
            if (dimi < head_dim) {
                o[qi * head_dim + dimi] = z[ri] / l;
            }
        }
    }
}

void check_inputs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v) {
    TORCH_CHECK(q.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "Q/K/V must have shape (S, d)");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q/K/V must have the same shape");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "Q/K/V must use float32");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "Q/K/V must have the same dtype");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q/K/V must be on the same CUDA device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0, "S and d must be positive");
    TORCH_CHECK(q.size(1) <= MAX_HEAD_DIM, "head dimension must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(q.numel() <= std::numeric_limits<int>::max(),
                "Q/K/V are too large for 32-bit kernel indexing");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() - KEYS_PER_TILE,
                "Sequence is too long for 32-bit tile indexing");
}

template <int DIMS_PER_LANE>
void launch_forward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                    const int seq_len, const int head_dim, const float scale, const bool is_causal,
                    const dim3 grid, const dim3 block, const cudaStream_t stream) {
    fa2_forward_fp32_kernel<DIMS_PER_LANE>
        <<<grid, block, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                     o.data_ptr<float>(), seq_len, head_dim, scale, is_causal);
}

constexpr std::array FORWARD_LAUNCHERS = {
    &launch_forward<1>,
    &launch_forward<2>,
    &launch_forward<3>,
    &launch_forward<4>,
};
static_assert(FORWARD_LAUNCHERS.size() == (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE);

} // namespace

at::Tensor fa2_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal) {
    check_inputs(q, k, v);
    const c10::cuda::CUDAGuard device_guard(q.device());

    const int seq_len = static_cast<int>(q.size(0));
    const int head_dim = static_cast<int>(q.size(1));
    auto o = at::empty_like(q);

    const dim3 grid(1 + (seq_len - 1) / WARPS_PER_BLOCK);
    const dim3 block(THREADS_PER_BLOCK);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int dims_per_lane = 1 + (head_dim - 1) / WARP_SIZE;
    FORWARD_LAUNCHERS[dims_per_lane - 1](
        q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream); // fmt: skip
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
