#pragma once

#include "maxwell_d64.cuh"
#include "score_bounds.cuh"

namespace fixed_pipeline {

constexpr int STAGE = 8;
constexpr unsigned MASK = 0xffffffffu;

struct alignas(16) Storage {
    float q[2][STAGE][64];
    float k[2][STAGE][64];
    float p[64][64];
    float v[2][STAGE][64];
    float denominator[2][64];
};

template <int ID>
__device__ __forceinline__ void group_sync() {
    asm volatile("bar.sync %0, 64;" : : "n"(ID) : "memory");
}

__device__ __forceinline__ void load(
    float (&out)[STAGE], const float *in, const int tid) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(in + tid * 4 + i * 64);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store(
    float (&out)[STAGE][64], const float (&in)[STAGE], const int tid) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const int e = tid * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

template <bool CAUSAL>
__device__ __forceinline__ void produce(
    Storage &shared, const float *q, const float *k, const int query_base, const int tile,
    const float scale, const float bias, float (&denominator)[8],
    const int tid, const int row, const int column, const bool final,
    float (&score)[8][8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            score[r][c] = 0.0f;
        }
    }
    float qg[STAGE], kg[STAGE];
    load(qg, q + query_base * 64, tid);
    load(kg, k + tile * 64, tid);
    store(shared.q[0], qg, tid);
    store(shared.k[0], kg, tid);
    group_sync<1>();
#pragma unroll 1
    for (int dimension = 0; dimension < 64; dimension += STAGE) {
        const int read = (dimension / STAGE) % 2;
        if (dimension + STAGE < 64) {
            load(qg, q + query_base * 64 + (dimension + STAGE) * 64, tid);
            load(kg, k + tile * 64 + (dimension + STAGE) * 64, tid);
        }
        maxwell_d64::pipeline_mma(score, shared.q[read], shared.k[read], row, column);
        if (dimension + STAGE < 64) {
            store(shared.q[read ^ 1], qg, tid);
            store(shared.k[read ^ 1], kg, tid);
            group_sync<1>();
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if constexpr (CAUSAL) {
                if (tile == query_base
                    && maxwell_d64::element(column, c) > maxwell_d64::element(row, r)) {
                    score[r][c] = -CUDART_INF_F;
                }
            }
            score[r][c] = maxwell_d64::exp2_scaled(score[r][c], scale, bias);
            total += score[r][c];
        }
        denominator[r] += total;
    }
    if (final) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float total = denominator[r];
            total += __shfl_xor_sync(MASK, total, 1);
            total += __shfl_xor_sync(MASK, total, 16);
            if ((tid & 17) == 0) {
                shared.denominator[tid / 32][maxwell_d64::element(row, r)] = total;
            }
        }
    }
}

__device__ __forceinline__ void consume(
    Storage &shared, const float *v, const int tile, float (&out)[8][8],
    const int tid, const int row, const int column) {
    float vg[STAGE];
    load(vg, v + tile * 64, tid);
    store(shared.v[0], vg, tid);
    group_sync<2>();
#pragma unroll 1
    for (int stage = 0; stage < 64; stage += STAGE) {
        const int read = (stage / STAGE) % 2;
        if (stage + STAGE < 64) {
            load(vg, v + (tile + stage + STAGE) * 64, tid);
        }
        maxwell_d64::pipeline_mma(
            out, *reinterpret_cast<const float (*)[STAGE][64]>(&shared.p[stage]),
            shared.v[read], row, column);
        if (stage + STAGE < 64) {
            store(shared.v[read ^ 1], vg, tid);
            group_sync<2>();
        }
    }
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
    const bool producer = threadIdx.x < 64;
    const int tid = threadIdx.x % 64;
    const int row = ((tid >> 1) & 7) * 4;
    const int column = (((tid & 0x30) >> 3) | (tid & 1)) * 4;
    const int query_base = blockIdx.x * 64;
    const int key_end = CAUSAL ? query_base + 64 : seq_len;
    const float bias = 64.0f - score_bounds::base(bounds, scale);
    float accumulator[8][8] = {};
    float denominator[8] = {};

    // Both roles use one loop and the same CTA barrier instructions on Pascal.
    for (int tile = 0; tile < key_end + 64; tile += 64) {
        if (producer && tile < key_end) {
            produce<CAUSAL>(shared, q, k, query_base, tile, scale, bias, denominator,
                            tid, row, column, tile + 64 == key_end, accumulator);
        } else if (!producer && tile) {
            consume(shared, v, tile - 64, accumulator, tid, row, column);
        }
        // Consumer has finished reading the only P buffer.
        __syncthreads();
        if (producer && tile < key_end) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ci = maxwell_d64::element(column, c);
#pragma unroll
                for (int r = 0; r < 8; r += 4) {
                    *reinterpret_cast<float4*>(&shared.p[ci][maxwell_d64::element(row, r)]) =
                        make_float4(accumulator[r][c], accumulator[r + 1][c],
                                    accumulator[r + 2][c], accumulator[r + 3][c]);
                }
            }
        }
        // Publish next P and, on the final producer iteration, its denominator.
        __syncthreads();
    }
    if (!producer) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int ri = maxwell_d64::element(row, r);
            const float inv = 1.0f / (shared.denominator[0][ri] + shared.denominator[1][ri]);
#pragma unroll
            for (int c = 0; c < 8; c += 4) {
                const int ci = maxwell_d64::element(column, c);
                *reinterpret_cast<float4*>(&o[(query_base + ri) * 64 + ci]) =
                    make_float4(accumulator[r][c] * inv, accumulator[r][c + 1] * inv,
                                accumulator[r][c + 2] * inv, accumulator[r][c + 3] * inv);
            }
        }
    }
}

} // namespace fixed_pipeline
