#pragma once

#include "register_mma.cuh"

namespace maxwell_d64 {

constexpr int THREADS = 32;
constexpr int TILE = 64;
constexpr int STAGE = 8;
constexpr unsigned MASK = 0xffffffffu;

__device__ __forceinline__ float exp2(const float x) {
    float result;
    asm("{\n"
        ".reg .f32 shifted, normal;\n"
        "add.rn.f32 shifted, %1, 0f42800000;\n"
        "ex2.approx.ftz.f32 normal, shifted;\n"
        "mul.rn.f32 %0, normal, 0f1f800000;\n"
        "}\n"
        : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ int element(const int base, const int i) {
    return base + (i / 4) * 32 + i % 4;
}

__device__ __forceinline__ int query_element(const int base, const int i) {
    return base + (i / 4) * 16 + i % 4;
}

template <int OFFSET>
__device__ __forceinline__ void fragment(float (&out)[8], const float *in, const int base) {
    const float4 lo = *reinterpret_cast<const float4*>(in + base);
    const float4 hi = *reinterpret_cast<const float4*>(in + base + OFFSET);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = lo.z;
    out[3] = lo.w;
    out[4] = hi.x;
    out[5] = hi.y;
    out[6] = hi.z;
    out[7] = hi.w;
}

__device__ __forceinline__ void load_qk(
    const float *q, const float *k, float (&qg)[8], float (&kg)[16],
    const int query_base, const int key_base, const int dimension) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        const float4 b = *reinterpret_cast<const float4*>(k + key_base * 64 + dimension * 64 + e);
        if (i < 8) {
            const int qe = threadIdx.x * 4 + i * THREADS;
            const int row = qe % 32;
            const int dim = dimension + qe / 32;
            const float4 a = *reinterpret_cast<const float4*>(
                q + (query_base / 64) * 4096 + dim * 64 + query_base % 64 + row);
            qg[i] = a.x;
            qg[i + 1] = a.y;
            qg[i + 2] = a.z;
            qg[i + 3] = a.w;
        }
        kg[i] = b.x;
        kg[i + 1] = b.y;
        kg[i + 2] = b.z;
        kg[i + 3] = b.w;
    }
}

__device__ __forceinline__ void store_qk(
    float (&qs)[8][36], float (&ks)[8][68],
    const float (&qg)[8], const float (&kg)[16]) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        if (i < 8) {
            *reinterpret_cast<float4*>(&qs[e / 32][e % 32]) =
                make_float4(qg[i], qg[i + 1], qg[i + 2], qg[i + 3]);
        }
        *reinterpret_cast<float4*>(&ks[e / 64][e % 64]) =
            make_float4(kg[i], kg[i + 1], kg[i + 2], kg[i + 3]);
    }
}

__device__ __forceinline__ void load_v(
    const float *v, float (&vg)[16], const int key_base) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        const float4 a = *reinterpret_cast<const float4*>(v + key_base * 64 + e);
        vg[i] = a.x;
        vg[i + 1] = a.y;
        vg[i + 2] = a.z;
        vg[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_v(
    float (&vs)[8][64], const float (&vg)[16]) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        *reinterpret_cast<float4*>(&vs[e / 64][e % 64]) =
            make_float4(vg[i], vg[i + 1], vg[i + 2], vg[i + 3]);
    }
}

template <int STRIDE_A, int STRIDE_B>
__device__ __forceinline__ void pipeline_mma(
    float (&c)[8][8], const float (&a_shared)[STAGE][STRIDE_A],
    const float (&b_shared)[STAGE][STRIDE_B], const int a_base, const int b_base) {
#pragma unroll 4
    for (int d = 0; d < STAGE; ++d) {
        float a[8], b[8];
        fragment<16>(a, a_shared[d], a_base);
        fragment<32>(b, b_shared[d], b_base);
        register_mma_8x8(c, a, b);
    }
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale) {
    struct alignas(16) Storage {
        union {
            struct {
                float q[2][8][36];
                float k[2][8][68];
            } qk;
            struct {
                float p[64][36];
                float v[2][STAGE][64];
            } pv;
        };
    };
    __shared__ Storage shared;
    const int lane = threadIdx.x % 32;
    const int query_fragment = ((lane >> 2) & 3) * 4;
    const int column_fragment = (((lane >> 4) * 4) | (lane & 3)) * 4;
    const int query_base = blockIdx.x * 32;
    const int key_end = CAUSAL ? (query_base / 64 + 1) * 64 : seq_len;
    float output[8][8] = {};
    float maximum[8];
    float denominator[8] = {};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        maximum[r] = -CUDART_INF_F;
    }

    for (int tile = 0; tile < key_end; tile += 64) {
        float score[8][8] = {};
        float qg[8], kg[16];
        load_qk(q, k, qg, kg, query_base, tile, 0);
        store_qk(shared.qk.q[0], shared.qk.k[0], qg, kg);
        __syncwarp(MASK);
#pragma unroll 1
        for (int dimension = 0; dimension < 64; dimension += 8) {
            const int read = (dimension / 8) % 2;
            if (dimension + 8 < 64) {
                load_qk(q, k, qg, kg, query_base, tile, dimension + 8);
            }
            pipeline_mma(score, shared.qk.q[read], shared.qk.k[read],
                         query_fragment, column_fragment);
            if (dimension + 8 < 64) {
                store_qk(shared.qk.q[read ^ 1], shared.qk.k[read ^ 1], qg, kg);
            }
            __syncwarp(MASK);
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                score[r][c] *= scale;
                if constexpr (CAUSAL) {
                    if (tile + 64 > query_base
                        && tile + element(column_fragment, c) > query_base + query_element(query_fragment, r)) {
                        score[r][c] = -CUDART_INF_F;
                    }
                }
                m = fmaxf(m, score[r][c]);
            }
            m = fmaxf(m, __shfl_xor_sync(MASK, m, 1));
            m = fmaxf(m, __shfl_xor_sync(MASK, m, 2));
            m = fmaxf(m, __shfl_xor_sync(MASK, m, 16));
            m = fmaxf(maximum[r], m);
            const float alpha = exp2(maximum[r] - m);
            maximum[r] = m;
            float total = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                score[r][c] = exp2(score[r][c] - m);
                total += score[r][c];
                output[r][c] *= alpha;
            }
            denominator[r] = denominator[r] * alpha + total;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int col = element(column_fragment, c);
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                const int row = query_element(query_fragment, r);
                *reinterpret_cast<float4*>(&shared.pv.p[col][row]) =
                    make_float4(score[r][c], score[r + 1][c], score[r + 2][c], score[r + 3][c]);
            }
        }
        float vg[16];
        load_v(v, vg, tile);
        store_v(shared.pv.v[0], vg);
        __syncwarp(MASK);
#pragma unroll 1
        for (int key_stage = 0; key_stage < 64; key_stage += STAGE) {
            const int read = (key_stage / STAGE) % 2;
            if (key_stage + STAGE < 64) {
                load_v(v, vg, tile + key_stage + STAGE);
            }
            pipeline_mma(output,
                         *reinterpret_cast<const float (*)[STAGE][36]>(&shared.pv.p[key_stage]),
                         shared.pv.v[read], query_fragment, column_fragment);
            if (key_stage + STAGE < 64) {
                store_v(shared.pv.v[read ^ 1], vg);
            }
            __syncwarp(MASK);
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(MASK, total, 1);
        total += __shfl_xor_sync(MASK, total, 2);
        total += __shfl_xor_sync(MASK, total, 16);
        denominator[r] = total;
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = query_base + query_element(query_fragment, r);
        const float inv = 1.0f / denominator[r];
#pragma unroll
        for (int c = 0; c < 8; c += 4) {
            const int col = element(column_fragment, c);
            *reinterpret_cast<float4*>(&o[row * 64 + col]) =
                make_float4(output[r][c] * inv, output[r][c + 1] * inv,
                            output[r][c + 2] * inv, output[r][c + 3] * inv);
        }
    }
}

} // namespace maxwell_d64
