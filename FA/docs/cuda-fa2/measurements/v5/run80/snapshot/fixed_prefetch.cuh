#pragma once

#include "fixed_wide.cuh"

namespace fixed_prefetch {

struct alignas(16) Storage {
    union {
        struct {
            float q[2][32][128];
            float k[64][64];
        } qk;
        struct {
            float p[64][128];
            float v[64][64];
        } pv;
    };
};

__device__ __forceinline__ void load_full(float (&out)[32], const float *in) {
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(in + threadIdx.x * 4 + i * 128);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_full(
    float (&out)[64][64], const float (&in)[32]) {
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const int e = threadIdx.x * 4 + i * 128;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

__device__ __forceinline__ void load_q(
    float (&out)[32], const float *in, const int base, const int dimension) {
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(
            in + (base + (threadIdx.x / 64) * 64) * 64
            + dimension * 64 + (threadIdx.x % 64) * 4 + i * 64);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_q(
    float (&out)[32][128], const float (&in)[32]) {
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const int e = (threadIdx.x % 64) * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64 + (threadIdx.x / 64) * 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

__device__ __forceinline__ void qk_mma(
    float (&c)[8][8], const float (&a)[32][128],
    const float (&b)[32][64], const int row, const int column) {
    float av[2][8], bv[2][8];
    maxwell_d64::fragment(av[0], a[0], row);
    maxwell_d64::fragment(bv[0], b[0], column);
#pragma unroll 2
    for (int dimension = 0; dimension < 32; dimension += 2) {
        maxwell_d64::fragment(av[1], a[dimension + 1], row);
        maxwell_d64::fragment(bv[1], b[dimension + 1], column);
        register_mma_8x8(c, av[0], bv[0]);
        if (dimension + 2 < 32) {
            maxwell_d64::fragment(av[0], a[dimension + 2], row);
            maxwell_d64::fragment(bv[0], b[dimension + 2], column);
        }
        register_mma_8x8(c, av[1], bv[1]);
    }
}

__device__ __forceinline__ void pv_mma(
    float (&c)[8][8], const float (&a)[64][128],
    const float (&b)[64][64], const int row, const int column) {
    float av[2][8], bv[2][8];
    maxwell_d64::fragment(av[0], a[0], row);
    maxwell_d64::fragment(bv[0], b[0], column);
#pragma unroll 4
    for (int dimension = 0; dimension < 64; dimension += 2) {
        maxwell_d64::fragment(av[1], a[dimension + 1], row);
        maxwell_d64::fragment(bv[1], b[dimension + 1], column);
        register_mma_8x8(c, av[0], bv[0]);
        if (dimension + 2 < 64) {
            maxwell_d64::fragment(av[0], a[dimension + 2], row);
            maxwell_d64::fragment(bv[0], b[dimension + 2], column);
        }
        register_mma_8x8(c, av[1], bv[1]);
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
    float qg[32], global[32];

    if (key_start < key_end) {
        load_full(global, k + key_start * 64);
        store_full(shared.qk.k, global);
        load_q(qg, q, query_base, 0);
    }
    for (int tile = key_start; tile < key_end; tile += 64) {
        float score[8][8] = {};
        store_q(shared.qk.q[0], qg);
        __syncthreads();
        load_q(qg, q, query_base, 32);
        qk_mma(score, shared.qk.q[0],
            *reinterpret_cast<const float (*)[32][64]>(&shared.qk.k[0]), row, column);
        store_q(shared.qk.q[1], qg);
        __syncthreads();
        load_full(global, v + tile * 64);
        qk_mma(score, shared.qk.q[1],
            *reinterpret_cast<const float (*)[32][64]>(&shared.qk.k[32]), row, column);
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
        store_full(shared.pv.v, global);
        __syncthreads();
        if (tile + 64 < key_end) {
            load_full(global, k + (tile + 64) * 64);
            load_q(qg, q, query_base, 0);
        }
        pv_mma(output, shared.pv.p, shared.pv.v, row, column);
        __syncthreads();
        if (tile + 64 < key_end) {
            store_full(shared.qk.k, global);
        }
    }
    float (*shared_denominator)[128] =
        reinterpret_cast<float (*)[128]>(&shared);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 16);
        if ((lane & 17) == 0) {
            shared_denominator[warp % 2][maxwell_d64::element(row, r)] = total;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int local_row = maxwell_d64::element(row, r);
        const float total = shared_denominator[0][local_row] + shared_denominator[1][local_row];
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

} // namespace fixed_prefetch
