#pragma once

#include "fixed_wide.cuh"

namespace fixed_long {

constexpr int STAGE = 8;

struct alignas(16) Storage {
    union {
        struct {
            float q[2][STAGE][128];
            float k[2][STAGE][128];
        } qk;
        struct {
            float p[64][128];
            float v[2][STAGE][64];
        } pv;
    };
    float denominator[2][128];
};

__device__ __forceinline__ void load_transposed(
    float (&out)[STAGE], const float *in, const int base, const int dimension) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(
            in + (base + (threadIdx.x / 64) * 64) * 64
            + dimension * 64 + (threadIdx.x % 64) * 4 + i * 64);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_transposed(
    float (&out)[STAGE][128], const float (&in)[STAGE]) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const int e = (threadIdx.x % 64) * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64 + (threadIdx.x / 64) * 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

__device__ __forceinline__ void load_v(float (&out)[STAGE / 2], const float *in) {
    const float4 a = *reinterpret_cast<const float4*>(in + threadIdx.x * 4);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
}

__device__ __forceinline__ void store_v(
    float (&out)[STAGE][64], const float (&in)[STAGE / 2]) {
    const int e = threadIdx.x * 4;
    *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
        make_float4(in[0], in[1], in[2], in[3]);
}

__device__ __forceinline__ void qk_mma(
    float (&c)[2][8][8], const float (&a)[STAGE][128],
    const float (&b)[STAGE][128], const int row, const int column) {
#pragma unroll 2
    for (int dimension = 0; dimension < STAGE; ++dimension) {
        float av[8], bv[2][8];
        maxwell_d64::fragment(av, a[dimension], row);
        maxwell_d64::fragment(bv[0], b[dimension], column);
        maxwell_d64::fragment(bv[1], b[dimension], column + 64);
        register_mma_8x8(c[0], av, bv[0]);
        register_mma_8x8(c[1], av, bv[1]);
    }
}

__device__ __forceinline__ void pv_mma(
    float (&c)[8][8], const float (&a)[STAGE][128],
    const float (&b)[STAGE][64], const int row, const int column) {
#pragma unroll 2
    for (int dimension = 0; dimension < STAGE; ++dimension) {
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

    for (int tile = key_start; tile < key_end; tile += 128) {
        float score[2][8][8] = {};
        float qg[STAGE], kg[STAGE];
        load_transposed(qg, q, query_base, 0);
        load_transposed(kg, k, tile, 0);
        store_transposed(shared.qk.q[0], qg);
        store_transposed(shared.qk.k[0], kg);
        __syncthreads();
#pragma unroll 1
        for (int dimension = 0; dimension < 64; dimension += STAGE) {
            const int read = (dimension / STAGE) % 2;
            if (dimension + STAGE < 64) {
                load_transposed(qg, q, query_base, dimension + STAGE);
                load_transposed(kg, k, tile, dimension + STAGE);
            }
            qk_mma(score, shared.qk.q[read], shared.qk.k[read], row, column);
            if (dimension + STAGE < 64) {
                store_transposed(shared.qk.q[read ^ 1], qg);
                store_transposed(shared.qk.k[read ^ 1], kg);
                __syncthreads();
            }
        }
        __syncthreads();

#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float total = 0.0f;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    if constexpr (CAUSAL) {
                        if (tile + half * 64 + maxwell_d64::element(column, c)
                            > query_base + maxwell_d64::element(row, r)) {
                            score[half][r][c] = -CUDART_INF_F;
                        }
                    }
                    score[half][r][c] = fixed_wide::exponent(score[half][r][c], scale, bias);
                    total += score[half][r][c];
                }
                denominator[r] += total;
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ci = maxwell_d64::element(column, c);
#pragma unroll
                for (int r = 0; r < 8; r += 4) {
                    *reinterpret_cast<float4*>(&shared.pv.p[ci][maxwell_d64::element(row, r)]) =
                        make_float4(score[half][r][c], score[half][r + 1][c],
                                    score[half][r + 2][c], score[half][r + 3][c]);
                }
            }
            float vg[STAGE / 2];
            load_v(vg, v + (tile + half * 64) * 64);
            store_v(shared.pv.v[0], vg);
            __syncthreads();
#pragma unroll 1
            for (int stage = 0; stage < 64; stage += STAGE) {
                const int read = (stage / STAGE) % 2;
                if (stage + STAGE < 64) {
                    load_v(vg, v + (tile + half * 64 + stage + STAGE) * 64);
                }
                pv_mma(output, *reinterpret_cast<const float (*)[STAGE][128]>(&shared.pv.p[stage]),
                       shared.pv.v[read], row, column);
                if (stage + STAGE < 64) {
                    store_v(shared.pv.v[read ^ 1], vg);
                    __syncthreads();
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 16);
        if ((lane & 17) == 0) {
            shared.denominator[warp % 2][maxwell_d64::element(row, r)] = total;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int local_row = maxwell_d64::element(row, r);
        const float total = shared.denominator[0][local_row] + shared.denominator[1][local_row];
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

} // namespace fixed_long
