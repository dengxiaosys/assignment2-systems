#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "register_mma.cuh"
#include "maxwell_d64.cuh"
#include "persistent_q.cuh"
#include "pack_qk.cuh"

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_QUERIES_PER_BLOCK = 128;
constexpr int SHARED_MEMORY_BUDGET = 48 * 1024;
constexpr int VECTOR_WIDTH = 4;
constexpr int MAX_HEAD_DIM = 128;
constexpr float LOG2_E = 1.4426950408889634074f;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

__device__ float softmax_exp2(const float value) {
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

template <int COLS>
__device__ float exponentiate_row(float (&scores)[COLS], const float maximum) {
    float total = 0.0f;
#pragma unroll
    for (int ci = 0; ci < COLS; ++ci) {
        scores[ci] = softmax_exp2(scores[ci] - maximum);
        total += scores[ci];
    }
    return total;
}

template <int ROWS, int COLS>
__device__ void outer_product(float (&accumulator)[ROWS][COLS], const float (&a)[ROWS],
                              const float (&b)[COLS]) {
    if constexpr (ROWS == 8 && COLS == 8) {
        register_mma_8x8(accumulator, a, b);
    } else {
#pragma unroll
    for (int coli = 0; coli < COLS; ++coli) {
#pragma unroll
        for (int rowi = 0; rowi < ROWS; ++rowi) {
            const int ri = coli % 2 == 0 ? rowi : ROWS - 1 - rowi;
            accumulator[ri][coli] += a[ri] * b[coli];
        }
    }
    }
}

template <int ROWS, int REDUCTION_STAGE, bool ALIGNED, bool BULK = false>
__device__ void load_transposed_stage(const float *input,
                                      float (&fragment)[ROWS * REDUCTION_STAGE / THREADS_PER_BLOCK],
                                      const int row_basei, const int row_limit,
                                      const int stagei, const int head_dim) {
    if constexpr (ALIGNED) {
#pragma unroll
        for (int fi = 0; fi < ROWS * REDUCTION_STAGE / THREADS_PER_BLOCK; fi += VECTOR_WIDTH) {
            const int elementi = threadIdx.x * VECTOR_WIDTH + fi * THREADS_PER_BLOCK;
            const int rowi = row_basei + elementi / REDUCTION_STAGE;
            const int dimi = stagei + elementi % REDUCTION_STAGE;
            float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (BULK || (rowi < row_limit && dimi < head_dim)) {
                values = *reinterpret_cast<const float4*>(&input[rowi * head_dim + dimi]);
            }
            fragment[fi] = values.x;
            fragment[fi + 1] = values.y;
            fragment[fi + 2] = values.z;
            fragment[fi + 3] = values.w;
        }
    } else {
#pragma unroll
        for (int fi = 0; fi < ROWS * REDUCTION_STAGE / THREADS_PER_BLOCK; ++fi) {
            const int elementi = threadIdx.x + fi * THREADS_PER_BLOCK;
            const int rowi = row_basei + elementi / REDUCTION_STAGE;
            const int dimi = stagei + elementi % REDUCTION_STAGE;
            fragment[fi] = rowi < row_limit && dimi < head_dim ? input[rowi * head_dim + dimi] : 0.0f;
        }
    }
}

template <int ROWS, int STRIDE, int REDUCTION_STAGE, bool ALIGNED>
__device__ void store_transposed_stage(float (&destination)[REDUCTION_STAGE][STRIDE],
                                       const float (&fragment)[ROWS * REDUCTION_STAGE
                                                               / THREADS_PER_BLOCK]) {
#pragma unroll
    for (int fi = 0; fi < ROWS * REDUCTION_STAGE / THREADS_PER_BLOCK; ++fi) {
        const int elementi = ALIGNED ? threadIdx.x * VECTOR_WIDTH
                                           + (fi / VECTOR_WIDTH) * THREADS_PER_BLOCK * VECTOR_WIDTH
                                           + fi % VECTOR_WIDTH
                                      : threadIdx.x + fi * THREADS_PER_BLOCK;
        destination[elementi % REDUCTION_STAGE][elementi / REDUCTION_STAGE] = fragment[fi];
    }
}

template <int ROWS, int STRIDE>
__device__ void load_shared_rows(float (&fragment)[ROWS], const float (&source)[STRIDE],
                                 const int rowi, const int chunk_stride) {
#pragma unroll
    for (int ri = 0; ri < ROWS; ri += VECTOR_WIDTH) {
        const float4 values = *reinterpret_cast<const float4*>(
            &source[rowi + (ri / VECTOR_WIDTH) * chunk_stride]);
        fragment[ri] = values.x;
        fragment[ri + 1] = values.y;
        fragment[ri + 2] = values.z;
        fragment[ri + 3] = values.w;
    }
}

template <int TILE_HEAD_DIM, int REDUCTION_STAGE, bool ALIGNED, bool BULK = false>
__device__ void load_value_stage(const float *v,
                                 float (&fragment)[REDUCTION_STAGE * TILE_HEAD_DIM / THREADS_PER_BLOCK],
                                 const int key_basei, const int key_limit, const int head_dim) {
    if constexpr (ALIGNED) {
#pragma unroll
        for (int fi = 0; fi < REDUCTION_STAGE * TILE_HEAD_DIM / THREADS_PER_BLOCK; fi += VECTOR_WIDTH) {
            const int elementi = threadIdx.x * VECTOR_WIDTH + fi * THREADS_PER_BLOCK;
            const int ki = key_basei + elementi / TILE_HEAD_DIM;
            const int dimi = elementi % TILE_HEAD_DIM;
            float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (BULK || (ki < key_limit && dimi < head_dim)) {
                values = *reinterpret_cast<const float4*>(&v[ki * head_dim + dimi]);
            }
            fragment[fi] = values.x;
            fragment[fi + 1] = values.y;
            fragment[fi + 2] = values.z;
            fragment[fi + 3] = values.w;
        }
    } else {
#pragma unroll
        for (int fi = 0; fi < REDUCTION_STAGE * TILE_HEAD_DIM / THREADS_PER_BLOCK; ++fi) {
            const int elementi = threadIdx.x + fi * THREADS_PER_BLOCK;
            const int ki = key_basei + elementi / TILE_HEAD_DIM;
            const int dimi = elementi % TILE_HEAD_DIM;
            fragment[fi] = ki < key_limit && dimi < head_dim ? v[ki * head_dim + dimi] : 0.0f;
        }
    }
}

template <int TILE_HEAD_DIM, int REDUCTION_STAGE, bool ALIGNED>
__device__ void store_value_stage(float (&destination)[REDUCTION_STAGE][TILE_HEAD_DIM],
                                  const float (&fragment)[REDUCTION_STAGE * TILE_HEAD_DIM
                                                          / THREADS_PER_BLOCK]) {
#pragma unroll
    for (int fi = 0; fi < REDUCTION_STAGE * TILE_HEAD_DIM / THREADS_PER_BLOCK; ++fi) {
        const int elementi = ALIGNED ? threadIdx.x * VECTOR_WIDTH
                                           + (fi / VECTOR_WIDTH) * THREADS_PER_BLOCK * VECTOR_WIDTH
                                           + fi % VECTOR_WIDTH
                                      : threadIdx.x + fi * THREADS_PER_BLOCK;
        destination[elementi / TILE_HEAD_DIM][elementi % TILE_HEAD_DIM] = fragment[fi];
    }
}

template <int LANES_PER_ROW>
__device__ int probability_row(const int queryi, const int keyi) {
    return queryi ^ (((keyi / VECTOR_WIDTH) % LANES_PER_ROW) * VECTOR_WIDTH);
}

template <int LANES_PER_ROW>
__device__ float row_max(float value) {
#pragma unroll
    for (int stride = LANES_PER_ROW / 2; stride > 0; stride /= 2) {
        value = fmaxf(value, __shfl_xor_sync(FULL_WARP_MASK, value, stride, LANES_PER_ROW));
    }
    return value;
}

template <int LANES_PER_ROW>
__device__ float row_sum(float value) {
#pragma unroll
    for (int stride = LANES_PER_ROW / 2; stride > 0; stride /= 2) {
        value += __shfl_xor_sync(FULL_WARP_MASK, value, stride, LANES_PER_ROW);
    }
    return value;
}

template <int DIMS_PER_LANE, bool ALIGNED, int HEAD_DIM = 0, int CAUSAL = -1, bool BULK = false>
__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                        const float *__restrict__ v, float *const __restrict__ o,
                                        const int seq_len, const int runtime_head_dim, const float scale,
                                        const bool runtime_is_causal) {
    const int head_dim = HEAD_DIM ? HEAD_DIM : runtime_head_dim;
    const bool is_causal = CAUSAL >= 0 ? static_cast<bool>(CAUSAL) : runtime_is_causal;
    constexpr int TILE_HEAD_DIM = ((DIMS_PER_LANE + 1) / 2) * 2 * WARP_SIZE;
    constexpr int LANES_PER_ROW = 8;
    constexpr int ROW_GROUPS = WARP_SIZE / LANES_PER_ROW;
    constexpr int KEYS_PER_TILE = 64;
    constexpr int ROWS_PER_THREAD = TILE_HEAD_DIM <= 64 ? 8 : 4;
    constexpr int QUERIES_PER_WARP = ROW_GROUPS * ROWS_PER_THREAD;
    constexpr int QUERIES_PER_BLOCK = QUERIES_PER_WARP * WARPS_PER_BLOCK;
    constexpr int REDUCTION_STAGE = TILE_HEAD_DIM <= 64 ? 16 : 8;
    constexpr int OUTPUTS_PER_THREAD = TILE_HEAD_DIM / LANES_PER_ROW;
    constexpr int KEY_STRIDE = KEYS_PER_TILE + VECTOR_WIDTH;
    constexpr int QUERY_STRIDE = QUERIES_PER_BLOCK + VECTOR_WIDTH;
    constexpr int KEYS_PER_THREAD = KEYS_PER_TILE / LANES_PER_ROW;
    constexpr int Q_GLOBAL_FRAGMENT = QUERIES_PER_BLOCK * REDUCTION_STAGE / THREADS_PER_BLOCK;
    constexpr int K_GLOBAL_FRAGMENT = KEYS_PER_TILE * REDUCTION_STAGE / THREADS_PER_BLOCK;
    constexpr int V_GLOBAL_FRAGMENT = REDUCTION_STAGE * TILE_HEAD_DIM / THREADS_PER_BLOCK;
    struct alignas(16) SharedStorage {
        union {
            struct {
                float q[2][REDUCTION_STAGE][QUERY_STRIDE];
                float k[2][REDUCTION_STAGE][KEY_STRIDE];
            } qk;
            struct {
                float p[KEYS_PER_TILE][QUERIES_PER_BLOCK];
                float v[2][REDUCTION_STAGE][TILE_HEAD_DIM];
            } pv;
        };
    };
    static_assert(sizeof(SharedStorage) <= SHARED_MEMORY_BUDGET);
    __shared__ SharedStorage shared;

    const int warpi = threadIdx.x / WARP_SIZE;
    const int lanei = threadIdx.x % WARP_SIZE;
    const int row_lanei = lanei / LANES_PER_ROW;
    const int col_lanei = lanei % LANES_PER_ROW;
    const int thread_queryi = warpi * QUERIES_PER_WARP + row_lanei * ROWS_PER_THREAD;
    const int thread_keyi = col_lanei * VECTOR_WIDTH;
    const int thread_dimi = col_lanei * VECTOR_WIDTH;
    const int query_basei = blockIdx.x * QUERIES_PER_BLOCK;
    const int block_key_limit = is_causal ? min(seq_len, query_basei + QUERIES_PER_BLOCK) : seq_len;

    float z[ROWS_PER_THREAD][OUTPUTS_PER_THREAD] = {};
    float m[ROWS_PER_THREAD];
    float l[ROWS_PER_THREAD] = {};
#pragma unroll
    for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
        m[qri] = -CUDART_INF_F;
    }

    for (int tilei = 0; tilei < block_key_limit; tilei += KEYS_PER_TILE) {
        float scores[ROWS_PER_THREAD][KEYS_PER_THREAD] = {};
        float q_global[Q_GLOBAL_FRAGMENT];
        float k_global[K_GLOBAL_FRAGMENT];
        load_transposed_stage<QUERIES_PER_BLOCK, REDUCTION_STAGE, ALIGNED, BULK>(
            q, q_global, query_basei, seq_len, 0, head_dim);
        load_transposed_stage<KEYS_PER_TILE, REDUCTION_STAGE, ALIGNED, BULK>(
            k, k_global, tilei, block_key_limit, 0, head_dim);
        store_transposed_stage<QUERIES_PER_BLOCK, QUERY_STRIDE, REDUCTION_STAGE, ALIGNED>(
            shared.qk.q[0], q_global);
        store_transposed_stage<KEYS_PER_TILE, KEY_STRIDE, REDUCTION_STAGE, ALIGNED>(
            shared.qk.k[0], k_global);
        __syncthreads();

#pragma unroll 1
        for (int stagei = 0; stagei < head_dim; stagei += REDUCTION_STAGE) {
            const int read_stagei = (stagei / REDUCTION_STAGE) % 2;
            const int write_stagei = read_stagei ^ 1;
            float q_values[ROWS_PER_THREAD];
            float k_values[KEYS_PER_THREAD];
            if (stagei + REDUCTION_STAGE < head_dim) {
                load_transposed_stage<QUERIES_PER_BLOCK, REDUCTION_STAGE, ALIGNED, BULK>(
                    q, q_global, query_basei, seq_len, stagei + REDUCTION_STAGE, head_dim);
                load_transposed_stage<KEYS_PER_TILE, REDUCTION_STAGE, ALIGNED, BULK>(
                    k, k_global, tilei, block_key_limit, stagei + REDUCTION_STAGE, head_dim);
            }
#pragma unroll 4
            for (int dimensioni = 0; dimensioni < REDUCTION_STAGE; ++dimensioni) {
                load_shared_rows(q_values, shared.qk.q[read_stagei][dimensioni],
                                 thread_queryi, VECTOR_WIDTH);
                load_shared_rows(k_values, shared.qk.k[read_stagei][dimensioni],
                                 thread_keyi, LANES_PER_ROW * VECTOR_WIDTH);
                outer_product(scores, q_values, k_values);
            }
            if (stagei + REDUCTION_STAGE < head_dim) {
                store_transposed_stage<QUERIES_PER_BLOCK, QUERY_STRIDE, REDUCTION_STAGE, ALIGNED>(
                    shared.qk.q[write_stagei], q_global);
                store_transposed_stage<KEYS_PER_TILE, KEY_STRIDE, REDUCTION_STAGE, ALIGNED>(
                    shared.qk.k[write_stagei], k_global);
            }
            __syncthreads();
        }

        const bool full_tile = (BULK || (query_basei + QUERIES_PER_BLOCK <= seq_len
            && tilei + KEYS_PER_TILE <= seq_len))
            && (!is_causal || tilei + KEYS_PER_TILE <= query_basei);
        if (!full_tile) {
#pragma unroll
            for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
                const int qi = query_basei + thread_queryi + qri;
                const int key_limit = qi < seq_len ? (is_causal ? qi + 1 : seq_len) : 0;
#pragma unroll
                for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                    const int ki = tilei + thread_keyi
                        + (kri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + kri % VECTOR_WIDTH;
                    if (ki >= key_limit) {
                        scores[qri][kri] = -CUDART_INF_F;
                    }
                }
            }
        }
#pragma unroll
        for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
            float maximum = -CUDART_INF_F;
#pragma unroll
            for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
                scores[qri][kri] *= scale;
                maximum = fmaxf(maximum, scores[qri][kri]);
            }
            const float m_new = fmaxf(m[qri], row_max<LANES_PER_ROW>(maximum));
            const float alpha = BULK ? softmax_exp2(m[qri] - m_new)
                : (isfinite(m[qri]) ? softmax_exp2(m[qri] - m_new) : 0.0f);
            const float l_tile = exponentiate_row(
                scores[qri], BULK ? m_new : (isfinite(m_new) ? m_new : 0.0f));
            l[qri] = l[qri] * alpha + row_sum<LANES_PER_ROW>(l_tile);
            m[qri] = m_new;
#pragma unroll
            for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
                z[qri][ri] *= alpha;
            }
        }

        // The final QK stage barrier finished every union reader before these stores.
#pragma unroll
        for (int kri = 0; kri < KEYS_PER_THREAD; ++kri) {
            const int keyi = thread_keyi
                + (kri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + kri % VECTOR_WIDTH;
#pragma unroll
            for (int qri = 0; qri < ROWS_PER_THREAD; qri += VECTOR_WIDTH) {
                const int queryi = probability_row<LANES_PER_ROW>(thread_queryi + qri, keyi);
                *reinterpret_cast<float4*>(&shared.pv.p[keyi][queryi]) = make_float4(
                    scores[qri][kri], scores[qri + 1][kri], scores[qri + 2][kri], scores[qri + 3][kri]);
            }
        }
        float v_global[V_GLOBAL_FRAGMENT];
        load_value_stage<TILE_HEAD_DIM, REDUCTION_STAGE, ALIGNED, BULK>(
            v, v_global, tilei, block_key_limit, head_dim);
        store_value_stage<TILE_HEAD_DIM, REDUCTION_STAGE, ALIGNED>(shared.pv.v[0], v_global);
        __syncthreads();

#pragma unroll 1
        for (int stagei = 0; stagei < KEYS_PER_TILE; stagei += REDUCTION_STAGE) {
            const int read_stagei = (stagei / REDUCTION_STAGE) % 2;
            const int write_stagei = read_stagei ^ 1;
            float p_values[ROWS_PER_THREAD];
            float v_values[OUTPUTS_PER_THREAD];
            if (stagei + REDUCTION_STAGE < KEYS_PER_TILE) {
                load_value_stage<TILE_HEAD_DIM, REDUCTION_STAGE, ALIGNED, BULK>(
                    v, v_global, tilei + stagei + REDUCTION_STAGE, block_key_limit, head_dim);
            }
#pragma unroll 4
            for (int keyi = 0; keyi < REDUCTION_STAGE; ++keyi) {
#pragma unroll
                for (int qri = 0; qri < ROWS_PER_THREAD; qri += VECTOR_WIDTH) {
                    const float4 probability = *reinterpret_cast<const float4*>(
                        &shared.pv.p[stagei + keyi][
                            probability_row<LANES_PER_ROW>(thread_queryi + qri, stagei + keyi)]);
                    p_values[qri] = probability.x;
                    p_values[qri + 1] = probability.y;
                    p_values[qri + 2] = probability.z;
                    p_values[qri + 3] = probability.w;
                }
                load_shared_rows(v_values, shared.pv.v[read_stagei][keyi],
                                 thread_dimi, LANES_PER_ROW * VECTOR_WIDTH);
                outer_product(z, p_values, v_values);
            }
            if (stagei + REDUCTION_STAGE < KEYS_PER_TILE) {
                store_value_stage<TILE_HEAD_DIM, REDUCTION_STAGE, ALIGNED>(
                    shared.pv.v[write_stagei], v_global);
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int qri = 0; qri < ROWS_PER_THREAD; ++qri) {
        const int qi = query_basei + thread_queryi + qri;
        if (BULK || qi < seq_len) {
            const float inverse_l = 1.0f / l[qri];
#pragma unroll
            for (int ri = 0; ri < OUTPUTS_PER_THREAD; ++ri) {
                const int dimi = thread_dimi
                    + (ri / VECTOR_WIDTH) * LANES_PER_ROW * VECTOR_WIDTH + ri % VECTOR_WIDTH;
                if (BULK || dimi < head_dim) {
                    o[qi * head_dim + dimi] = z[qri][ri] * inverse_l;
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
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() - MAX_QUERIES_PER_BLOCK,
                "Sequence is too long for 32-bit tile indexing");
}

template <int DIMS_PER_LANE>
void launch_forward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                    const int seq_len, const int head_dim, const float scale, const bool is_causal,
                    const dim3 grid, const dim3 block, const cudaStream_t stream) {
    const uintptr_t input_addresses = reinterpret_cast<uintptr_t>(q.data_ptr<float>())
        | reinterpret_cast<uintptr_t>(k.data_ptr<float>())
        | reinterpret_cast<uintptr_t>(v.data_ptr<float>());
    if (head_dim % VECTOR_WIDTH == 0 && input_addresses % alignof(float4) == 0) {
        if constexpr (DIMS_PER_LANE == 2) {
            if (head_dim == 64 && seq_len % 128 == 0) {
                const dim3 maxwell_grid(seq_len / 64);
                auto packed = at::empty({2, seq_len, 64}, q.options());
                packed_qk::prepare<<<dim3(seq_len / 64, 2, 4), dim3(32, 8), 0, stream>>>(
                    q.data_ptr<float>(), k.data_ptr<float>(), packed.data_ptr<float>());
                const float *packed_q = packed.data_ptr<float>();
                const float *packed_k = packed_q + seq_len * 64;
                auto bounds = at::zeros({3}, q.options().dtype(at::kInt));
                score_bounds::prepare<<<dim3((seq_len * 16 + 255) / 256), 256, 0, stream>>>(
                    q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                    reinterpret_cast<unsigned*>(bounds.data_ptr<int>()), seq_len * 64);
                const float *score_bounds = reinterpret_cast<const float*>(bounds.data_ptr<int>());
                if (is_causal) {
                    maxwell_d64::forward<true>
                        <<<maxwell_grid, 64, 0, stream>>>(packed_q, packed_k,
                            v.data_ptr<float>(), o.data_ptr<float>(), seq_len, scale, score_bounds);
                } else {
                    maxwell_d64::forward<false>
                        <<<maxwell_grid, 64, 0, stream>>>(packed_q, packed_k,
                            v.data_ptr<float>(), o.data_ptr<float>(), seq_len, scale, score_bounds);
                }
                return;
            }
        }
        fa2_forward_fp32_kernel<DIMS_PER_LANE, true>
            <<<grid, block, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                         o.data_ptr<float>(), seq_len, head_dim, scale, is_causal);
    } else {
        fa2_forward_fp32_kernel<DIMS_PER_LANE, false>
            <<<grid, block, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                         o.data_ptr<float>(), seq_len, head_dim, scale, is_causal);
    }
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

    const int queries_per_block = head_dim <= 64 ? 128 : 64;
    const dim3 grid(1 + (seq_len - 1) / queries_per_block);
    const dim3 block(THREADS_PER_BLOCK);
    const float scale = LOG2_E / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int dims_per_lane = 1 + (head_dim - 1) / WARP_SIZE;
    FORWARD_LAUNCHERS[dims_per_lane - 1](
        q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream); // fmt: skip
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
