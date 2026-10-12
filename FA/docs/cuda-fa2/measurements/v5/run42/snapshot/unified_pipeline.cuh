#pragma once

#include "register_mma.cuh"

namespace unified_pipeline {

constexpr unsigned MASK = 0xffffffffu;

struct alignas(16) Storage {
    float q[2][8][68];
    float k[2][8][68];
    float p[64][64];
    float v[2][8][68];
    float maximum[2][64];
    float denominator[2][64];
    float alpha[64];
};

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

__device__ __forceinline__ void fragment(float (&out)[8], const float *in) {
    const float4 lo = *reinterpret_cast<const float4*>(in);
    const float4 hi = *reinterpret_cast<const float4*>(in + 32);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = lo.z;
    out[3] = lo.w;
    out[4] = hi.x;
    out[5] = hi.y;
    out[6] = hi.z;
    out[7] = hi.w;
}

__device__ __forceinline__ void mma(
    float (&c)[8][8], const float *as, const float *bs, const int stride_a) {
#pragma unroll 4
    for (int d = 0; d < 8; ++d) {
        float a[8], b[8];
        fragment(a, as + d * stride_a);
        fragment(b, bs + d * 68);
        register_mma_8x8(c, a, b);
    }
}

__device__ __forceinline__ void load(float (&out)[8], const float *in, const int tid) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const float4 value = *reinterpret_cast<const float4*>(in + tid * 4 + i * 64);
        out[i] = value.x;
        out[i + 1] = value.y;
        out[i + 2] = value.z;
        out[i + 3] = value.w;
    }
}

__device__ __forceinline__ void store(
    float (&out)[8][68], const float (&in)[8], const int tid) {
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const int e = tid * 4 + i * 64;
        *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
            make_float4(in[i], in[i + 1], in[i + 2], in[i + 3]);
    }
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale) {
    __shared__ Storage shared;
    const bool producer = threadIdx.x < 64;
    const int tid = threadIdx.x % 64;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row = ((tid >> 1) & 7) * 4;
    const int column = (((tid & 0x30) >> 3) | (tid & 1)) * 4;
    const int query_base = blockIdx.x * 64;
    const int key_end = CAUSAL ? query_base + 64 : seq_len;
    float accumulator[8][8] = {};
    float maximum[8], denominator[8] = {};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        maximum[r] = -CUDART_INF_F;
    }

    for (int tile = 0; tile < key_end + 64; tile += 64) {
        const bool active = producer ? tile < key_end : tile > 0;
        if (producer) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    accumulator[r][c] = 0.0f;
                }
            }
        } else if (active) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float alpha = shared.alpha[element(row, r)];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    accumulator[r][c] *= alpha;
                }
            }
        }

        const float *global_a = q + query_base * 64;
        const float *global_b = producer ? k + min(tile, key_end - 64) * 64
                                        : v + max(tile - 64, 0) * 64;
        float (*shared_b)[8][68] = producer ? shared.k : shared.v;
        float ag[8], bg[8];
        if (active) {
            if (producer) {
                load(ag, global_a, tid);
                store(shared.q[0], ag, tid);
            }
            load(bg, global_b, tid);
            store(shared_b[0], bg, tid);
        }
        __syncthreads();

#pragma unroll 1
        for (int stage = 0; stage < 64; stage += 8) {
            const int read = stage / 8 % 2;
            if (active) {
                if (stage + 8 < 64) {
                    if (producer) {
                        load(ag, global_a + (stage + 8) * 64, tid);
                    }
                    load(bg, global_b + (stage + 8) * 64, tid);
                }
                const float *as = producer ? &shared.q[read][0][row]
                                           : &shared.p[stage][row];
                const float *bs = &shared_b[read][0][column];
                mma(accumulator, as, bs, producer ? 68 : 64);
                if (stage + 8 < 64) {
                    if (producer) {
                        store(shared.q[read ^ 1], ag, tid);
                    }
                    store(shared_b[read ^ 1], bg, tid);
                }
            }
            __syncthreads();
        }

        if (producer && active) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float m = -CUDART_INF_F;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    accumulator[r][c] *= scale;
                    if constexpr (CAUSAL) {
                        if (tile == query_base && element(column, c) > element(row, r)) {
                            accumulator[r][c] = -CUDART_INF_F;
                        }
                    }
                    m = fmaxf(m, accumulator[r][c]);
                }
                m = fmaxf(m, __shfl_xor_sync(MASK, m, 1));
                m = fmaxf(m, __shfl_xor_sync(MASK, m, 16));
                if ((lane & 17) == 0) {
                    shared.maximum[warp][element(row, r)] = m;
                }
            }
        }
        __syncthreads();
        if (producer && active) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int ri = element(row, r);
                const float m = fmaxf(maximum[r],
                    fmaxf(shared.maximum[0][ri], shared.maximum[1][ri]));
                const float alpha = exp2(maximum[r] - m);
                maximum[r] = m;
                if (warp == 0 && (lane & 17) == 0) {
                    shared.alpha[ri] = alpha;
                }
                float total = 0.0f;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    accumulator[r][c] = exp2(accumulator[r][c] - m);
                    total += accumulator[r][c];
                }
                denominator[r] = denominator[r] * alpha + total;
                if (tile + 64 == key_end) {
                    float sum = denominator[r];
                    sum += __shfl_xor_sync(MASK, sum, 1);
                    sum += __shfl_xor_sync(MASK, sum, 16);
                    if ((lane & 17) == 0) {
                        shared.denominator[warp][ri] = sum;
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ci = element(column, c);
#pragma unroll
                for (int r = 0; r < 8; r += 4) {
                    *reinterpret_cast<float4*>(&shared.p[ci][element(row, r)]) =
                        make_float4(accumulator[r][c], accumulator[r + 1][c],
                                    accumulator[r + 2][c], accumulator[r + 3][c]);
                }
            }
        }
        __syncthreads();
    }

    if (!producer) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int ri = element(row, r);
            const float inv = 1.0f / (shared.denominator[0][ri] + shared.denominator[1][ri]);
#pragma unroll
            for (int c = 0; c < 8; c += 4) {
                *reinterpret_cast<float4*>(&o[(query_base + ri) * 64 + element(column, c)]) =
                    make_float4(accumulator[r][c] * inv, accumulator[r][c + 1] * inv,
                                accumulator[r][c + 2] * inv, accumulator[r][c + 3] * inv);
            }
        }
    }
}

} // namespace unified_pipeline
