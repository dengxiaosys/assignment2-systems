#pragma once

#include "register_mma.cuh"

namespace maxwell_d64 {

constexpr int THREADS = 64;
constexpr int TILE = 64;
constexpr int STAGE = 8;
constexpr unsigned MASK = 0xffffffffu;
constexpr float RESCALE_THRESHOLD = 8.0f;

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

__device__ __forceinline__ float exp2_scaled(
    const float score, const float scale, const float bias) {
    float result;
    asm("{\n"
        ".reg .f32 shifted, normal;\n"
        "fma.rn.f32 shifted, %1, %2, %3;\n"
        "ex2.approx.ftz.f32 normal, shifted;\n"
        "mul.rn.f32 %0, normal, 0f1f800000;\n"
        "}\n" : "=f"(result) : "f"(score), "f"(scale), "f"(bias));
    return result;
}

__device__ __forceinline__ int element(const int base, const int i) {
    return base + (i / 4) * 32 + i % 4;
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

__device__ __forceinline__ void load_qk(
    const float *q, const float *k, float (&qg)[8], float (&kg)[8],
    const int query_base, const int key_base, const int dimension) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        const float4 a = *reinterpret_cast<const float4*>(q + query_base * 64 + dimension * 64 + e);
        const float4 b = *reinterpret_cast<const float4*>(k + key_base * 64 + dimension * 64 + e);
        qg[i] = a.x;
        qg[i + 1] = a.y;
        qg[i + 2] = a.z;
        qg[i + 3] = a.w;
        kg[i] = b.x;
        kg[i + 1] = b.y;
        kg[i + 2] = b.z;
        kg[i + 3] = b.w;
    }
}

__device__ __forceinline__ void store_qk(
    float (&qs)[8][68], float (&ks)[8][68], const float (&qg)[8], const float (&kg)[8]) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        *reinterpret_cast<float4*>(&qs[e / 64][e % 64]) =
            make_float4(qg[i], qg[i + 1], qg[i + 2], qg[i + 3]);
        *reinterpret_cast<float4*>(&ks[e / 64][e % 64]) =
            make_float4(kg[i], kg[i + 1], kg[i + 2], kg[i + 3]);
    }
}

__device__ __forceinline__ void load_v(
    const float *v, float (&vg)[8], const int key_base) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        const float4 a = *reinterpret_cast<const float4*>(v + key_base * 64 + e);
        vg[i] = a.x;
        vg[i + 1] = a.y;
        vg[i + 2] = a.z;
        vg[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_v(float (&vs)[8][64], const float (&vg)[8]) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        *reinterpret_cast<float4*>(&vs[e / 64][e % 64]) =
            make_float4(vg[i], vg[i + 1], vg[i + 2], vg[i + 3]);
    }
}

template <int STRIDE_A, int STRIDE_B>
__device__ __forceinline__ void pipeline_mma(
    float (&c)[8][8], const float (&a_shared)[8][STRIDE_A],
    const float (&b_shared)[8][STRIDE_B], const int a_base, const int b_base) {
#pragma unroll 4
    for (int d = 0; d < 8; ++d) {
        float a[8], b[8];
        fragment(a, a_shared[d], a_base);
        fragment(b, b_shared[d], b_base);
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
                float q[2][8][68];
                float k[2][8][68];
            } qk;
            struct {
                float p[64][68];
                float v[2][8][64];
            } pv;
        };
        float maximum[2][64];
        float denominator[2][64];
    };
    __shared__ Storage shared;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int query_fragment = ((threadIdx.x >> 1) & 7) * 4;
    const int column_fragment = (((threadIdx.x & 0x30) >> 3) | (threadIdx.x & 1)) * 4;
    const int query_base = blockIdx.x * 64;
    const int key_end = CAUSAL ? query_base + 64 : seq_len;
    float output[8][8] = {};
    float maximum[8];
    float denominator[8] = {};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        maximum[r] = -CUDART_INF_F;
    }

    for (int tile = 0; tile < key_end; tile += 64) {
        float score[8][8] = {};
        float qg[8], kg[8];
        load_qk(q, k, qg, kg, query_base, tile, 0);
        store_qk(shared.qk.q[0], shared.qk.k[0], qg, kg);
        __syncthreads();
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
            if (dimension + 8 < 64) {
                __syncthreads();
            }
        }
        float tile_maximum[8];
        bool need_rescale = false;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if constexpr (CAUSAL) {
                    if (tile == query_base
                        && tile + element(column_fragment, c) > query_base + element(query_fragment, r)) {
                        score[r][c] = -CUDART_INF_F;
                    }
                }
                m = fmaxf(m, score[r][c]);
            }
            m *= scale;
            tile_maximum[r] = m;
            need_rescale |= m > maximum[r] + RESCALE_THRESHOLD;
        }
        const bool rescale_tile = __syncthreads_or(need_rescale);
        if (rescale_tile) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float m = tile_maximum[r];
                m = fmaxf(m, __shfl_xor_sync(MASK, m, 1));
                m = fmaxf(m, __shfl_xor_sync(MASK, m, 16));
                if ((lane & 17) == 0) {
                    shared.maximum[warp][element(query_fragment, r)] = m;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = element(query_fragment, r);
            const float candidate = rescale_tile
                ? fmaxf(shared.maximum[0][row], shared.maximum[1][row]) : maximum[r];
            const bool rescale = rescale_tile && candidate > maximum[r] + RESCALE_THRESHOLD;
            const float m = rescale ? candidate : maximum[r];
            const float alpha = rescale ? exp2(maximum[r] - m) : 1.0f;
            maximum[r] = m;
            const float bias = 64.0f - m;
            float total = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                score[r][c] = exp2_scaled(score[r][c], scale, bias);
                total += score[r][c];
            }
            if (rescale) {
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    output[r][c] *= alpha;
                }
                denominator[r] *= alpha;
            }
            denominator[r] += total;
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int col = element(column_fragment, c);
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                const int row = element(query_fragment, r);
                *reinterpret_cast<float4*>(&shared.pv.p[col][row]) =
                    make_float4(score[r][c], score[r + 1][c], score[r + 2][c], score[r + 3][c]);
            }
        }
        float vg[8];
        load_v(v, vg, tile);
        store_v(shared.pv.v[0], vg);
        __syncthreads();
#pragma unroll 1
        for (int key_stage = 0; key_stage < 64; key_stage += 8) {
            const int read = (key_stage / 8) % 2;
            if (key_stage + 8 < 64) {
                load_v(v, vg, tile + key_stage + 8);
            }
            pipeline_mma(output,
                         *reinterpret_cast<const float (*)[8][68]>(&shared.pv.p[key_stage]),
                         shared.pv.v[read], query_fragment, column_fragment);
            if (key_stage + 8 < 64) {
                store_v(shared.pv.v[read ^ 1], vg);
            }
            if (key_stage + 8 < 64) {
                __syncthreads();
            }
        }
        // Final PV readers must finish before the next tile reuses the union.
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float total = denominator[r];
        total += __shfl_xor_sync(MASK, total, 1);
        total += __shfl_xor_sync(MASK, total, 16);
        if ((lane & 17) == 0) {
            shared.denominator[warp][element(query_fragment, r)] = total;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = query_base + element(query_fragment, r);
        const int local_row = element(query_fragment, r);
        const float inv = 1.0f / (shared.denominator[0][local_row] + shared.denominator[1][local_row]);
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
