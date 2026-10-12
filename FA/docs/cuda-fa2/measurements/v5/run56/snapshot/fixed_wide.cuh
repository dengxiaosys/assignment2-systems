#pragma once

#include "maxwell_d64.cuh"

namespace fixed_wide {

constexpr int STAGE = 16;
constexpr int THREADS = 128;

__device__ __forceinline__ float exponent(const float score, const float scale, const float bias) {
    float result;
    asm("{\n"
        ".reg .f32 x;\n"
        "fma.rn.f32 x, %1, %2, %3;\n"
        "ex2.approx.ftz.f32 %0, x;\n"
        "}\n" : "=f"(result) : "f"(score), "f"(scale), "f"(bias));
    return result;
}

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

__device__ __forceinline__ void load_kv(float (&out)[STAGE / 2], const float *in) {
#pragma unroll
    for (int i = 0; i < STAGE / 2; i += 4) {
        const float4 a = *reinterpret_cast<const float4*>(in + threadIdx.x * 4 + i * THREADS);
        out[i] = a.x;
        out[i + 1] = a.y;
        out[i + 2] = a.z;
        out[i + 3] = a.w;
    }
}

__device__ __forceinline__ void store_kv(
    float (&out)[STAGE][64], const float (&in)[STAGE / 2]) {
#pragma unroll
    for (int i = 0; i < STAGE / 2; i += 4) {
        const int e = threadIdx.x * 4 + i * THREADS;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

template <int A_STRIDE>
__device__ __forceinline__ void mma(
    float (&c)[8][8], const float (&a)[STAGE][A_STRIDE],
    const float (&b)[STAGE][64], const int row, const int column) {
#pragma unroll 4
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

    for (int tile = key_start; tile < key_end; tile += 64) {
        float score[8][8] = {};
        float qg[STAGE], kg[STAGE / 2];
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
            mma(score, shared.qk.q[read], shared.qk.k[read], row, column);
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
                score[r][c] = exponent(score[r][c], scale, bias);
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
        float vg[STAGE / 2];
        load_kv(vg, v + tile * 64);
        store_kv(shared.pv.v[0], vg);
        __syncthreads();
#pragma unroll 1
        for (int stage = 0; stage < 64; stage += STAGE) {
            const int read = (stage / STAGE) % 2;
            if (stage + STAGE < 64) {
                load_kv(vg, v + (tile + stage + STAGE) * 64);
            }
            mma(
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
        const float total = shared.denominator[0][local_row] + shared.denominator[1][local_row];
        const int ri = query_base + maxwell_d64::element(row, r);
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

__global__ void combine(
    const float *numerator, const float *denominator, float *out,
    const int seq_len, const int splits, const float *bounds, const float scale) {
    if (!score_bounds::eligible(bounds, scale)) {
        return;
    }
    const int e = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (e < seq_len * 64) {
        float4 total = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float d = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float4 a = *reinterpret_cast<const float4*>(numerator + split * seq_len * 64 + e);
            total.x += a.x;
            total.y += a.y;
            total.z += a.z;
            total.w += a.w;
            d += denominator[split * seq_len + e / 64];
        }
        const float inv = 1.0f / d;
        *reinterpret_cast<float4*>(out + e) =
            make_float4(total.x * inv, total.y * inv, total.z * inv, total.w * inv);
    }
}

} // namespace fixed_wide
