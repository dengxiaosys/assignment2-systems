#pragma once

#include "maxwell_d64.cuh"

namespace fixed_wide {

constexpr int STAGE = 8;
constexpr int THREADS = 128;

struct alignas(16) Storage {
    union {
        struct {
            float q[2][STAGE][128];
            float k[2][STAGE][64];
        } qk;
        struct {
            float p[64][128];
            float v[2][STAGE][64];
        } pv;
    };
    float denominator[2][128];
};

__device__ __forceinline__ void load_q(
    float (&out)[STAGE], const float *in, const int query_base, const int dimension) {
    const int half = threadIdx.x / 64;
    const int tid = threadIdx.x % 64;
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(
            in + (query_base + half * 64) * 64 + dimension * 64 + tid * 4 + i * 64);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_q(
    float (&out)[STAGE][128], const float (&in)[STAGE]) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const int e = (threadIdx.x % 64) * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64 + (threadIdx.x / 64) * 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

__device__ __forceinline__ void load_kv(float (&out)[4], const float *in) {
    const float4 a = *reinterpret_cast<const float4*>(in + threadIdx.x * 4);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
}

__device__ __forceinline__ void store_kv(float (&out)[STAGE][64], const float (&in)[4]) {
    const int e = threadIdx.x * 4;
    *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
        make_float4(in[0], in[1], in[2], in[3]);
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale,
    const float *bounds) {
    if (!score_bounds::eligible(bounds, scale)) {
        return;
    }
    __shared__ Storage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = (threadIdx.x / 64) * 64 + ((threadIdx.x >> 1) & 7) * 4;
    const int column = (((threadIdx.x & 0x30) >> 3) | (threadIdx.x & 1)) * 4;
    const int query_base = blockIdx.x * 128;
    const int key_end = CAUSAL ? query_base + 128 : seq_len;
    const float bias = 64.0f - score_bounds::base(bounds, scale);
    float output[8][8] = {};
    float denominator[8] = {};

    for (int tile = 0; tile < key_end; tile += 64) {
        float score[8][8] = {};
        float qg[STAGE], kg[4];
        load_q(qg, q, query_base, 0);
        load_kv(kg, k + tile * 64);
        store_q(shared.qk.q[0], qg);
        store_kv(shared.qk.k[0], kg);
        __syncthreads();
#pragma unroll 1
        for (int dimension = 0; dimension < 64; dimension += STAGE) {
            const int read = (dimension / STAGE) % 2;
            if (dimension + STAGE < 64) {
                load_q(qg, q, query_base, dimension + STAGE);
                load_kv(kg, k + tile * 64 + (dimension + STAGE) * 64);
            }
            maxwell_d64::pipeline_mma(score, shared.qk.q[read], shared.qk.k[read], row, column);
            if (dimension + STAGE < 64) {
                store_q(shared.qk.q[read ^ 1], qg);
                store_kv(shared.qk.k[read ^ 1], kg);
                __syncthreads();
            }
        }
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
                score[r][c] = maxwell_d64::exp2_scaled(score[r][c], scale, bias);
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
        float vg[4];
        load_kv(vg, v + tile * 64);
        store_kv(shared.pv.v[0], vg);
        __syncthreads();
#pragma unroll 1
        for (int stage = 0; stage < 64; stage += STAGE) {
            const int read = (stage / STAGE) % 2;
            if (stage + STAGE < 64) {
                load_kv(vg, v + (tile + stage + STAGE) * 64);
            }
            maxwell_d64::pipeline_mma(
                output, *reinterpret_cast<const float (*)[STAGE][128]>(&shared.pv.p[stage]),
                shared.pv.v[read], row, column);
            if (stage + STAGE < 64) {
                store_kv(shared.pv.v[read ^ 1], vg);
                __syncthreads();
            }
        }
        __syncthreads();
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
        const float inv = 1.0f / (shared.denominator[0][local_row] + shared.denominator[1][local_row]);
        const int ri = query_base + maxwell_d64::element(row, r);
#pragma unroll
        for (int c = 0; c < 8; c += 4) {
            const int ci = maxwell_d64::element(column, c);
            *reinterpret_cast<float4*>(&o[ri * 64 + ci]) =
                make_float4(output[r][c] * inv, output[r][c + 1] * inv,
                            output[r][c + 2] * inv, output[r][c + 3] * inv);
        }
    }
}

} // namespace fixed_wide
