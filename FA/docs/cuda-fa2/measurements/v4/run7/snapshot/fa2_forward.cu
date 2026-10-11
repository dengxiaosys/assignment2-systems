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
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_KEYS_PER_TILE = WARP_SIZE;
constexpr int SHARED_MEMORY_BUDGET = 48 * 1024;
constexpr int LANES_PER_ROW = 8;
constexpr int VECTOR_WIDTH = 4;
constexpr int ROW_GROUPS = WARP_SIZE / LANES_PER_ROW;
constexpr int ROWS_PER_THREAD = 4;
constexpr int QUERIES_PER_WARP = ROW_GROUPS * ROWS_PER_THREAD;
constexpr int QUERIES_PER_BLOCK = QUERIES_PER_WARP * WARPS_PER_BLOCK;
constexpr int MAX_HEAD_DIM = 128;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
static_assert(MAX_KEYS_PER_TILE % LANES_PER_ROW == 0);

__device__ float row_max(float value) {
    for (int stride = LANES_PER_ROW / 2; stride > 0; stride /= 2) {
        value = fmaxf(value, __shfl_xor_sync(FULL_WARP_MASK, value, stride, LANES_PER_ROW));
    }
    return value;
}

__device__ float row_sum(float value) {
    for (int stride = LANES_PER_ROW / 2; stride > 0; stride /= 2) {
        value += __shfl_xor_sync(FULL_WARP_MASK, value, stride, LANES_PER_ROW);
    }
    return value;
}

template <int DIMS_PER_LANE>
__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                        const float *__restrict__ v, float *const __restrict__ o,
                                        const int seq_len, const int head_dim, const float scale,
                                        const bool is_causal) {
    constexpr int TILE_HEAD_DIM = DIMS_PER_LANE * WARP_SIZE;
    constexpr int OUTPUTS_PER_THREAD = TILE_HEAD_DIM / LANES_PER_ROW;
    constexpr int max_keys_by_storage =
        (SHARED_MEMORY_BUDGET / sizeof(float) - QUERIES_PER_BLOCK * LANES_PER_ROW)
        / (QUERIES_PER_BLOCK + TILE_HEAD_DIM);
    constexpr int capped_keys =
        max_keys_by_storage < MAX_KEYS_PER_TILE ? max_keys_by_storage : MAX_KEYS_PER_TILE;
    constexpr int KEYS_PER_TILE = capped_keys / WARP_SIZE * WARP_SIZE;
    constexpr int KEY_STRIDE = KEYS_PER_TILE + VECTOR_WIDTH;
    constexpr int KEYS_PER_THREAD = KEYS_PER_TILE / LANES_PER_ROW;
    constexpr int PROBABILITY_STRIDE = KEYS_PER_TILE + LANES_PER_ROW;
    static_assert(KEYS_PER_TILE >= WARP_SIZE);
    union alignas(16) SharedStorage {
        float k[TILE_HEAD_DIM][KEY_STRIDE];
        struct {
            float p[QUERIES_PER_BLOCK][PROBABILITY_STRIDE];
            float v[KEYS_PER_TILE][TILE_HEAD_DIM];
        } pv;
    };
    static_assert(sizeof(SharedStorage) <= SHARED_MEMORY_BUDGET);
    __shared__ SharedStorage shared;

    const int warpi = threadIdx.x / WARP_SIZE;
    const int lanei = threadIdx.x % WARP_SIZE;
    const int row_lanei = lanei / LANES_PER_ROW;
    const int col_lanei = lanei % LANES_PER_ROW;
    const int query_basei = blockIdx.x * QUERIES_PER_BLOCK;
    const int block_key_limit = is_causal ? min(seq_len, query_basei + QUERIES_PER_BLOCK) : seq_len;

    float q_dim[ROWS_PER_THREAD][OUTPUTS_PER_THREAD];
    float z[ROWS_PER_THREAD][OUTPUTS_PER_THREAD] = {};
    float m[ROWS_PER_THREAD];
    float l[ROWS_PER_THREAD] = {};
#pragma unroll
    for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
        const int queryi = warpi * QUERIES_PER_WARP + row_lanei + qri * ROW_GROUPS;
        const int qi = query_basei + queryi;
        m[qri] = -CUDART_INF_F;
#pragma unroll
        for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
            const int dimi = col_lanei * VECTOR_WIDTH
                + (ri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + ri % VECTOR_WIDTH;
            q_dim[qri][ri] = qi < seq_len && dimi < head_dim ? q[qi * head_dim + dimi] : 0.0f;
        }
    }

    for (int tilei = 0; tilei < block_key_limit; tilei += KEYS_PER_TILE) {
        for (int elementi = threadIdx.x; elementi < KEYS_PER_TILE * TILE_HEAD_DIM;
             elementi += THREADS_PER_BLOCK) {
            const int keyi = elementi / TILE_HEAD_DIM;
            const int dimi = elementi % TILE_HEAD_DIM;
            const int ki = tilei + keyi;
            shared.k[dimi][keyi] =
                ki < block_key_limit && dimi < head_dim ? k[ki * head_dim + dimi] : 0.0f;
        }
        __syncthreads(); // Publish K to every query warp.

        float scores[ROWS_PER_THREAD][KEYS_PER_THREAD] = {};
#pragma unroll
        for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
#pragma unroll 1
            for (int dim_lanei = 0; dim_lanei < LANES_PER_ROW; ++dim_lanei) {
                const int dimi = (ri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH
                    + dim_lanei * VECTOR_WIDTH + ri % VECTOR_WIDTH;
                float q_values[ROWS_PER_THREAD];
                float k_values[KEYS_PER_THREAD];
#pragma unroll
                for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
                    q_values[qri] = __shfl_sync(
                        FULL_WARP_MASK, q_dim[qri][ri], dim_lanei, LANES_PER_ROW);
                }
#pragma unroll
                for (int kri = 0; kri < KEYS_PER_THREAD; kri += VECTOR_WIDTH) {
                    const float4 values = *reinterpret_cast<const float4*>(
                        &shared.k[dimi][col_lanei * VECTOR_WIDTH + kri * LANES_PER_ROW]);
                    k_values[kri] = values.x;
                    k_values[kri + 1] = values.y;
                    k_values[kri + 2] = values.z;
                    k_values[kri + 3] = values.w;
                }
#pragma unroll
                for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
#pragma unroll
                    for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                        scores[qri][kri] += q_values[qri] * k_values[kri];
                    }
                }
            }
        }

#pragma unroll
        for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
            const int queryi = warpi * QUERIES_PER_WARP + row_lanei + qri * ROW_GROUPS;
            const int qi = query_basei + queryi;
            const int key_limit = qi < seq_len ? (is_causal ? qi + 1 : seq_len) : 0;
            float maximum = -CUDART_INF_F;
#pragma unroll
            for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                const int ki = tilei + col_lanei * VECTOR_WIDTH
                    + (kri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + kri % VECTOR_WIDTH;
                scores[qri][kri] = ki < key_limit ? scores[qri][kri] * scale : -CUDART_INF_F;
                maximum = fmaxf(maximum, scores[qri][kri]);
            }
            const float m_new = fmaxf(m[qri], row_max(maximum));
            const float alpha = isfinite(m[qri]) ? __expf(m[qri] - m_new) : 0.0f;
            float l_tile = 0.0f;
#pragma unroll
            for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                const int ki = tilei + col_lanei * VECTOR_WIDTH
                    + (kri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + kri % VECTOR_WIDTH;
                const float p_tilde = ki < key_limit ? __expf(scores[qri][kri] - m_new) : 0.0f;
                scores[qri][kri] = p_tilde;
                l_tile += p_tilde;
            }
            l[qri] = alpha * l[qri] + row_sum(l_tile);
            m[qri] = m_new;
#pragma unroll
            for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
                z[qri][ri] *= alpha;
            }
        }
        __syncthreads(); // Finish K readers before overwriting the union with V.

#pragma unroll
        for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
            const int queryi = warpi * QUERIES_PER_WARP + row_lanei + qri * ROW_GROUPS;
#pragma unroll
            for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                const int keyi = col_lanei * VECTOR_WIDTH
                    + (kri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + kri % VECTOR_WIDTH;
                shared.pv.p[queryi][keyi] = scores[qri][kri];
            }
        }
        for (int elementi = threadIdx.x; elementi < KEYS_PER_TILE * TILE_HEAD_DIM;
             elementi += THREADS_PER_BLOCK) {
            const int keyi = elementi / TILE_HEAD_DIM;
            const int dimi = elementi % TILE_HEAD_DIM;
            const int ki = tilei + keyi;
            shared.pv.v[keyi][dimi] =
                ki < block_key_limit && dimi < head_dim ? v[ki * head_dim + dimi] : 0.0f;
        }
        __syncthreads(); // Publish P and V after their cooperative writes.

#pragma unroll 2
        for (int keyi = 0; keyi < KEYS_PER_TILE; ++keyi) {
            float p_values[ROWS_PER_THREAD];
            float v_values[OUTPUTS_PER_THREAD];
#pragma unroll
            for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
                const int queryi = warpi * QUERIES_PER_WARP + row_lanei + qri * ROW_GROUPS;
                p_values[qri] = shared.pv.p[queryi][keyi];
            }
#pragma unroll
            for (int ri = 0; ri < OUTPUTS_PER_THREAD; ri += VECTOR_WIDTH) {
                const float4 values = *reinterpret_cast<const float4*>(
                    &shared.pv.v[keyi][col_lanei * VECTOR_WIDTH + ri * LANES_PER_ROW]);
                v_values[ri] = values.x;
                v_values[ri + 1] = values.y;
                v_values[ri + 2] = values.z;
                v_values[ri + 3] = values.w;
            }
#pragma unroll
            for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
#pragma unroll
                for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
                    z[qri][ri] += p_values[qri] * v_values[ri];
                }
            }
        }
        __syncthreads(); // Finish all V readers before the next K tile.
    }

#pragma unroll
    for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
        const int queryi = warpi * QUERIES_PER_WARP + row_lanei + qri * ROW_GROUPS;
        const int qi = query_basei + queryi;
        if (qi < seq_len) {
#pragma unroll
            for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
                const int dimi = col_lanei * VECTOR_WIDTH
                    + (ri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + ri % VECTOR_WIDTH;
                if (dimi < head_dim) {
                    o[qi * head_dim + dimi] = z[qri][ri] / l[qri];
                }
            }
        }
    }
}

void check_inputs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "Q/K/V must have shape (S, d)");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q/K/V must have the same shape");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat
                    && v.scalar_type() == at::kFloat, "Q/K/V must use float32");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q/K/V must be on the same CUDA device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0, "S and d must be positive");
    TORCH_CHECK(q.size(1) <= MAX_HEAD_DIM, "head dimension must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(q.numel() <= std::numeric_limits<int>::max(),
                "Q/K/V are too large for 32-bit kernel indexing");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() - QUERIES_PER_BLOCK,
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
    &launch_forward<1>, &launch_forward<2>, &launch_forward<3>, &launch_forward<4>,
};
static_assert(FORWARD_LAUNCHERS.size() == (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE);

} // namespace

at::Tensor fa2_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal) {
    check_inputs(q, k, v);
    const c10::cuda::CUDAGuard device_guard(q.device());

    const int seq_len = static_cast<int>(q.size(0));
    const int head_dim = static_cast<int>(q.size(1));
    auto o = at::empty_like(q);

    const dim3 grid(1 + (seq_len - 1) / QUERIES_PER_BLOCK);
    const dim3 block(THREADS_PER_BLOCK);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int dims_per_lane = 1 + (head_dim - 1) / WARP_SIZE;
    FORWARD_LAUNCHERS[dims_per_lane - 1](
        q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream); // fmt: skip
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
