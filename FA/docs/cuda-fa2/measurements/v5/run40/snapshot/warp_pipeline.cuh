#pragma once

#include "register_mma.cuh"

namespace warp_pipeline {

constexpr unsigned MASK = 0xffffffffu;
constexpr int STAGE = 8;

struct alignas(16) Storage {
    float q[2][STAGE][68];
    float k[2][STAGE][68];
    float p[1][64][64];
    float v[2][STAGE][64];
    float maximum[2][64];
    float denominator[2][64];
    float alpha[1][64];
};

template <int ID, int COUNT = 64>
__device__ __forceinline__ void group_sync() {
    asm volatile("bar.sync %0, %1;" : : "n"(ID), "n"(COUNT) : "memory");
}

template <int ID>
__device__ __forceinline__ void arrive() {
    asm volatile("bar.arrive %0, 128;" : : "n"(ID) : "memory");
}

__device__ __forceinline__ int element(const int base, const int i) {
    return base + (i / 4) * 32 + i % 4;
}

__device__ __forceinline__ float exp2(const float x) {
    float result;
    asm("{\n"
        ".reg .f32 shifted, normal;\n"
        "add.rn.f32 shifted, %1, 0f42800000;\n"
        "ex2.approx.ftz.f32 normal, shifted;\n"
        "mul.rn.f32 %0, normal, 0f1f800000;\n"
        "}\n" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ void fragment(float (&out)[8], const float *in, const int base) {
    const float4 lo = *reinterpret_cast<const float4*>(in + base);
    const float4 hi = *reinterpret_cast<const float4*>(in + base + 32);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = lo.z;
    out[3] = lo.w;
    out[4] = hi.x;
    out[5] = hi.y;
    out[6] = hi.z;
    out[7] = hi.w;
}

template <int STRIDE_A, int STRIDE_B>
__device__ __forceinline__ void mma(
    float (&c)[8][8], const float (&as)[STAGE][STRIDE_A],
    const float (&bs)[STAGE][STRIDE_B], const int row, const int column) {
#pragma unroll 4
    for (int d = 0; d < STAGE; ++d) {
        float a[8], b[8];
        fragment(a, as[d], row);
        fragment(b, bs[d], column);
        register_mma_8x8(c, a, b);
    }
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

template <int STRIDE>
__device__ __forceinline__ void store(
    float (&out)[STAGE][STRIDE], const float (&in)[STAGE], const int tid) {
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
    const float scale, float (&maximum)[8], float (&denominator)[8],
    const int tid, const int row, const int column, const bool final,
    float (&score)[8][8], float (&scaling)[8]) {
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
        mma(score, shared.q[read], shared.k[read], row, column);
        if (dimension + STAGE < 64) {
            store(shared.q[read ^ 1], qg, tid);
            store(shared.k[read ^ 1], kg, tid);
        }
        group_sync<1>();
    }
    const int warp = tid / 32;
    const int lane = tid % 32;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            score[r][c] *= scale;
            if constexpr (CAUSAL) {
                if (tile == query_base && element(column, c) > element(row, r)) {
                    score[r][c] = -CUDART_INF_F;
                }
            }
            m = fmaxf(m, score[r][c]);
        }
        m = fmaxf(m, __shfl_xor_sync(MASK, m, 1));
        m = fmaxf(m, __shfl_xor_sync(MASK, m, 16));
        if ((lane & 17) == 0) {
            shared.maximum[warp][element(row, r)] = m;
        }
    }
    group_sync<1>();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int ri = element(row, r);
        const float m = fmaxf(maximum[r], fmaxf(shared.maximum[0][ri], shared.maximum[1][ri]));
        const float alpha = exp2(maximum[r] - m);
        scaling[r] = alpha;
        maximum[r] = m;
        float total = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            score[r][c] = exp2(score[r][c] - m);
            total += score[r][c];
        }
        denominator[r] = denominator[r] * alpha + total;
    }
    if (final) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float total = denominator[r];
            total += __shfl_xor_sync(MASK, total, 1);
            total += __shfl_xor_sync(MASK, total, 16);
            if ((lane & 17) == 0) {
                shared.denominator[warp][element(row, r)] = total;
            }
        }
    }
}

__device__ __forceinline__ void consume(
    Storage &shared, const float *v, const int tile, float (&out)[8][8],
    const int tid, const int row, const int column) {
    const int buffer = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float alpha = shared.alpha[buffer][element(row, r)];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            out[r][c] *= alpha;
        }
    }
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
        mma(out, *reinterpret_cast<const float (*)[STAGE][64]>(&shared.p[buffer][stage]),
            shared.v[read], row, column);
        if (stage + STAGE < 64) {
            store(shared.v[read ^ 1], vg, tid);
        }
        group_sync<2>();
    }
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale) {
    __shared__ Storage shared;
    const bool producer = threadIdx.x < 64;
    const int tid = threadIdx.x % 64;
    const int row = ((tid >> 1) & 7) * 4;
    const int column = (((tid & 0x30) >> 3) | (tid & 1)) * 4;
    const int query_base = blockIdx.x * 64;
    const int key_end = CAUSAL ? query_base + 64 : seq_len;
    if (producer) {
        float accumulator[8][8] = {};
        float maximum[8], denominator[8] = {};
        float scaling[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            maximum[r] = -CUDART_INF_F;
        }
        for (int tile = 0; tile < key_end; tile += 64) {
            produce<CAUSAL>(shared, q, k, query_base, tile, scale, maximum,
                            denominator, tid, row, column, tile + 64 == key_end,
                            accumulator, scaling);
            // Consumer arrives once at startup, then once per consumed tile.
            // Producer waits before overwriting P/alpha and arrives after publish.
            group_sync<3, 128>();
            if (tid / 32 == 0 && (tid & 17) == 0) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    shared.alpha[0][element(row, r)] = scaling[r];
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ci = element(column, c);
#pragma unroll
                for (int r = 0; r < 8; r += 4) {
                    *reinterpret_cast<float4*>(&shared.p[0][ci][element(row, r)]) =
                        make_float4(accumulator[r][c], accumulator[r + 1][c],
                                    accumulator[r + 2][c], accumulator[r + 3][c]);
                }
            }
            arrive<4>();
        }
        // Keep producer warps live until the last buffer has been consumed.
        group_sync<3, 128>();
    } else {
        float accumulator[8][8] = {};
        arrive<3>();
        for (int tile = 0; tile < key_end; tile += 64) {
            group_sync<4, 128>();
            consume(shared, v, tile, accumulator, tid, row, column);
            arrive<3>();
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int ri = element(row, r);
            const float inv = 1.0f / (shared.denominator[0][ri] + shared.denominator[1][ri]);
#pragma unroll
            for (int c = 0; c < 8; c += 4) {
                const int ci = element(column, c);
                *reinterpret_cast<float4*>(&o[(query_base + ri) * 64 + ci]) =
                    make_float4(accumulator[r][c] * inv, accumulator[r][c + 1] * inv,
                                accumulator[r][c + 2] * inv, accumulator[r][c + 3] * inv);
            }
        }
    }
}

} // namespace warp_pipeline
