#pragma once

#include "fixed_wide.cuh"

namespace full_wide {

struct alignas(16) Storage {
    union {
        struct {
            float q[64][128];
            float k[64][64];
        } qk;
        struct {
            float p[64][128];
            float v[64][64];
        } pv;
    };
};
static_assert(sizeof(Storage) == 48 * 1024);

__device__ __forceinline__ void load_q(float (&out)[64][128], const float *in) {
    const int half = threadIdx.x / 64;
    const int tid = threadIdx.x % 64;
#pragma unroll 1
    for (int i = 0; i < 16; ++i) {
        const int e = tid * 4 + i * 256;
        const float4 a = *reinterpret_cast<const float4*>(in + half * 4096 + e);
        *reinterpret_cast<float4*>(&out[e / 64][half * 64 + e % 64]) = a;
    }
}

__device__ __forceinline__ void load_kv(float (&out)[64][64], const float *in) {
#pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        const int e = threadIdx.x * 4 + i * 512;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
            *reinterpret_cast<const float4*>(in + e);
    }
}

__device__ __forceinline__ void mma(
    float (&c)[8][8], const float (&a)[64][128],
    const float (&b)[64][64], const int row, const int column) {
#pragma unroll 4
    for (int dimension = 0; dimension < 64; ++dimension) {
        float av[8], bv[8];
        maxwell_d64::fragment(av, a[dimension], row);
        maxwell_d64::fragment(bv, b[dimension], column);
        register_mma_8x8(c, av, bv);
    }
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale,
    const float *bounds, float *partial_denominator, const int splits) {
    if (!score_bounds::eligible(bounds, scale)) {
        return;
    }
    __shared__ Storage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = (threadIdx.x / 64) * 64 + ((threadIdx.x >> 1) & 7) * 4;
    const int column = (((threadIdx.x & 0x30) >> 3) | (threadIdx.x & 1)) * 4;
    const int query_base = blockIdx.x * 128;
    const int chunk = seq_len / splits;
    const int key_start = blockIdx.y * chunk;
    const int key_end = min(key_start + chunk, CAUSAL ? query_base + 128 : seq_len);
    const float bias = -score_bounds::base(bounds, scale);
    float output[8][8] = {};
    float denominator[8] = {};

    for (int tile = key_start; tile < key_end; tile += 64) {
        float score[8][8] = {};
        load_q(shared.qk.q, q + query_base * 64);
        load_kv(shared.qk.k, k + tile * 64);
        __syncthreads();
        mma(score, shared.qk.q, shared.qk.k, row, column);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float total = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if constexpr (CAUSAL) {
                    if (tile + maxwell_d64::element(column, c)
                        > query_base + maxwell_d64::element(row, r)) {
                        score[r][c] = -CUDART_INF_F;
                    }
                }
                score[r][c] = fixed_wide::exponent(score[r][c], scale, bias);
                total += score[r][c];
            }
            denominator[r] += total;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int ci = maxwell_d64::element(column, c);
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                *reinterpret_cast<float4*>(&shared.pv.p[ci][maxwell_d64::element(row, r)]) =
                    make_float4(score[r][c], score[r + 1][c], score[r + 2][c], score[r + 3][c]);
            }
        }
        load_kv(shared.pv.v, v + tile * 64);
        __syncthreads();
        mma(output, shared.pv.p, shared.pv.v, row, column);
        __syncthreads();
    }
    auto &sums = *reinterpret_cast<float (*)[2][128]>(&shared.qk.q[0]);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 16);
        if ((lane & 17) == 0) {
            sums[warp % 2][maxwell_d64::element(row, r)] = total;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int local_row = maxwell_d64::element(row, r);
        const float total = sums[0][local_row] + sums[1][local_row];
        const int ri = query_base + local_row;
        if (threadIdx.x % 64 < 16 && !(lane & 1)) {
            partial_denominator[blockIdx.y * seq_len + ri] = total;
        }
#pragma unroll
        for (int c = 0; c < 8; c += 4) {
            const int ci = maxwell_d64::element(column, c);
            *reinterpret_cast<float4*>(&o[(blockIdx.y * seq_len + ri) * 64 + ci]) =
                make_float4(output[r][c], output[r][c + 1], output[r][c + 2], output[r][c + 3]);
        }
    }
}

} // namespace full_wide
