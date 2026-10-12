#pragma once

#include "fixed_wide.cuh"

namespace wide_pipeline {

constexpr int STAGE = 8;

struct alignas(16) Storage {
    float q[2][STAGE][128];
    float k[2][STAGE][64];
    float p[64][128];
    float v[2][STAGE][64];
};
static_assert(sizeof(Storage) == 48 * 1024);

template <int ID>
__device__ __forceinline__ void group_sync() {
    asm volatile("bar.sync %0, 64;" : : "n"(ID) : "memory");
}

template <int COUNT>
__device__ __forceinline__ void load(float (&out)[COUNT], const float *in, const int tid) {
#pragma unroll
    for (int i = 0; i < COUNT; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(in + tid * 4 + i * 64);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

template <int STRIDE>
__device__ __forceinline__ void store(
    float (&out)[STAGE][STRIDE], const float (&in)[STAGE], const int tid,
    const int half = 0) {
#pragma unroll
    for (int i = 0; i < STAGE; i += 4) {
        const int e = tid * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][half * 64 + e % 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

__device__ __forceinline__ void load_q(
    float (&out)[2][STAGE], const float *q, const int query_base,
    const int dimension, const int tid) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        load(out[half], q + (query_base + half * 64) * 64 + dimension * 64, tid);
    }
}

__device__ __forceinline__ void store_q(
    float (&out)[STAGE][128], const float (&in)[2][STAGE], const int tid) {
    store(out, in[0], tid, 0);
    store(out, in[1], tid, 1);
}

__device__ __forceinline__ void update(
    float (&c)[16][8], const float (&a)[16], const float (&b)[8]) {
    register_mma_8x8(*reinterpret_cast<float (*)[8][8]>(&c[0]),
                     *reinterpret_cast<const float (*)[8]>(&a[0]), b);
    register_mma_8x8(*reinterpret_cast<float (*)[8][8]>(&c[8]),
                     *reinterpret_cast<const float (*)[8]>(&a[8]), b);
}

__device__ __forceinline__ void mma(
    float (&c)[16][8], const float (&a)[STAGE][128],
    const float (&b)[STAGE][64], const int row, const int column) {
#pragma unroll 4
    for (int d = 0; d < STAGE; ++d) {
        float av[16], bv[8];
        maxwell_d64::fragment(*reinterpret_cast<float (*)[8]>(&av[0]), a[d], row);
        maxwell_d64::fragment(*reinterpret_cast<float (*)[8]>(&av[8]), a[d], row + 64);
        maxwell_d64::fragment(bv, b[d], column);
        update(c, av, bv);
    }
}

template <bool CAUSAL>
__device__ __forceinline__ void produce(
    Storage &shared, const float *q, const float *k, const int query_base,
    const int tile, const float scale, const float bias, float (&denominator)[16],
    const int tid, const int row, const int column, const bool final,
    float (&score)[16][8]) {
#pragma unroll
    for (int r = 0; r < 16; ++r) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            score[r][c] = 0.0f;
        }
    }
    float qg[2][STAGE], kg[STAGE];
    load_q(qg, q, query_base, 0, tid);
    load(kg, k + tile * 64, tid);
    store_q(shared.q[0], qg, tid);
    store(shared.k[0], kg, tid);
    group_sync<1>();
#pragma unroll 1
    for (int dimension = 0; dimension < 64; dimension += STAGE) {
        const int read = (dimension / STAGE) % 2;
        if (dimension + STAGE < 64) {
            load_q(qg, q, query_base, dimension + STAGE, tid);
            load(kg, k + tile * 64 + (dimension + STAGE) * 64, tid);
        }
        mma(score, shared.q[read], shared.k[read], row, column);
        if (dimension + STAGE < 64) {
            store_q(shared.q[read ^ 1], qg, tid);
            store(shared.k[read ^ 1], kg, tid);
            group_sync<1>();
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
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
    if (final) {
        group_sync<1>();
        auto &sums = *reinterpret_cast<float (*)[2][128]>(&shared.q[0]);
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float total = denominator[r];
            total += __shfl_xor_sync(0xffffffffu, total, 1);
            total += __shfl_xor_sync(0xffffffffu, total, 16);
            if ((tid & 17) == 0) {
                sums[tid / 32][maxwell_d64::element(row, r)] = total;
            }
        }
    }
}

__device__ __forceinline__ void consume(
    Storage &shared, const float *v, const int tile, float (&out)[16][8],
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
        mma(out, *reinterpret_cast<const float (*)[STAGE][128]>(&shared.p[stage]),
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
    const float *bounds, float *partial_denominator, const int splits) {
    if (!score_bounds::eligible(bounds, scale)) {
        return;
    }
    __shared__ Storage shared;
    const bool producer = threadIdx.x < 64;
    const int tid = threadIdx.x % 64;
    const int row = ((tid >> 1) & 7) * 4;
    const int column = (((tid & 0x30) >> 3) | (tid & 1)) * 4;
    const int query_base = blockIdx.x * 128;
    const int chunk = seq_len / splits;
    const int key_start = blockIdx.y * chunk;
    const int key_end = min(key_start + chunk, CAUSAL ? query_base + 128 : seq_len);
    if (key_start >= key_end) {
        // Consumers own all output elements and one column group owns each sum.
        if (!producer) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int ri = query_base + maxwell_d64::element(row, r);
                if (tid < 16 && !(tid & 1)) {
                    partial_denominator[blockIdx.y * seq_len + ri] = 0.0f;
                }
#pragma unroll
                for (int c = 0; c < 8; c += 4) {
                    const int ci = maxwell_d64::element(column, c);
                    *reinterpret_cast<float4*>(&o[(blockIdx.y * seq_len + ri) * 64 + ci]) =
                        make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        return;
    }
    const float bias = -score_bounds::base(bounds, scale);
    float accumulator[16][8] = {};
    float denominator[16] = {};

    for (int tile = key_start; tile < key_end + 64; tile += 64) {
        if (producer && tile < key_end) {
            produce<CAUSAL>(shared, q, k, query_base, tile, scale, bias, denominator,
                            tid, row, column, tile + 64 == key_end, accumulator);
        } else if (!producer && tile > key_start) {
            consume(shared, v, tile - 64, accumulator, tid, row, column);
        }
        __syncthreads();
        if (producer && tile < key_end) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ci = maxwell_d64::element(column, c);
#pragma unroll
                for (int r = 0; r < 16; r += 4) {
                    *reinterpret_cast<float4*>(&shared.p[ci][maxwell_d64::element(row, r)]) =
                        make_float4(accumulator[r][c], accumulator[r + 1][c],
                                    accumulator[r + 2][c], accumulator[r + 3][c]);
                }
            }
        }
        __syncthreads();
    }
    if (!producer) {
        const auto &sums = *reinterpret_cast<const float (*)[2][128]>(&shared.q[0]);
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int local_row = maxwell_d64::element(row, r);
            const int ri = query_base + local_row;
            if (tid < 16 && !(tid & 1)) {
                partial_denominator[blockIdx.y * seq_len + ri] =
                    sums[0][local_row] + sums[1][local_row];
            }
#pragma unroll
            for (int c = 0; c < 8; c += 4) {
                const int ci = maxwell_d64::element(column, c);
                *reinterpret_cast<float4*>(&o[(blockIdx.y * seq_len + ri) * 64 + ci]) =
                    make_float4(accumulator[r][c], accumulator[r][c + 1],
                                accumulator[r][c + 2], accumulator[r][c + 3]);
            }
        }
    }
}

} // namespace wide_pipeline
