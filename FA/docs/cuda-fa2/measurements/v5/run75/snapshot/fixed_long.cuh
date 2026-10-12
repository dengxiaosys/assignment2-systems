#pragma once

#include "fixed_wide.cuh"

namespace fixed_long {

constexpr int STAGE = 8;

#define LONG_FMA_PAIR(C00, C01, C10, C11, A0, A1, B0, B1) \
    "fma.rn.f32 %" #C00 ", %" #A0 ", %" #B0 ", %" #C00 ";\n" \
    "fma.rn.f32 %" #C10 ", %" #A1 ", %" #B0 ", %" #C10 ";\n" \
    "fma.rn.f32 %" #C11 ", %" #A1 ", %" #B1 ", %" #C11 ";\n" \
    "fma.rn.f32 %" #C01 ", %" #A0 ", %" #B1 ", %" #C01 ";\n"

#define LONG_ACC_ROW(H, R) \
    "+f"(c[H][R][0]), "+f"(c[H][R][1]), "+f"(c[H][R][2]), "+f"(c[H][R][3]), \
    "+f"(c[H][R][4]), "+f"(c[H][R][5]), "+f"(c[H][R][6]), "+f"(c[H][R][7])

__device__ __forceinline__ void register_mma_8x16(
    float (&c)[2][8][8], const float (&a)[8], const float (&b)[2][8]) {
    asm(
        LONG_FMA_PAIR(16, 17, 0, 1, 130, 128, 136, 137)
        LONG_FMA_PAIR(24, 25, 8, 9, 131, 129, 136, 137)
        LONG_FMA_PAIR(48, 49, 32, 33, 134, 132, 136, 137)
        LONG_FMA_PAIR(56, 57, 40, 41, 135, 133, 136, 137)
        LONG_FMA_PAIR(58, 59, 42, 43, 135, 133, 138, 139)
        LONG_FMA_PAIR(50, 51, 34, 35, 134, 132, 138, 139)
        LONG_FMA_PAIR(26, 27, 10, 11, 131, 129, 138, 139)
        LONG_FMA_PAIR(18, 19, 2, 3, 130, 128, 138, 139)
        LONG_FMA_PAIR(20, 21, 4, 5, 130, 128, 140, 141)
        LONG_FMA_PAIR(28, 29, 12, 13, 131, 129, 140, 141)
        LONG_FMA_PAIR(52, 53, 36, 37, 134, 132, 140, 141)
        LONG_FMA_PAIR(60, 61, 44, 45, 135, 133, 140, 141)
        LONG_FMA_PAIR(62, 63, 46, 47, 135, 133, 142, 143)
        LONG_FMA_PAIR(54, 55, 38, 39, 134, 132, 142, 143)
        LONG_FMA_PAIR(30, 31, 14, 15, 131, 129, 142, 143)
        LONG_FMA_PAIR(22, 23, 6, 7, 130, 128, 142, 143)
        LONG_FMA_PAIR(80, 81, 64, 65, 130, 128, 144, 145)
        LONG_FMA_PAIR(88, 89, 72, 73, 131, 129, 144, 145)
        LONG_FMA_PAIR(112, 113, 96, 97, 134, 132, 144, 145)
        LONG_FMA_PAIR(120, 121, 104, 105, 135, 133, 144, 145)
        LONG_FMA_PAIR(122, 123, 106, 107, 135, 133, 146, 147)
        LONG_FMA_PAIR(114, 115, 98, 99, 134, 132, 146, 147)
        LONG_FMA_PAIR(90, 91, 74, 75, 131, 129, 146, 147)
        LONG_FMA_PAIR(82, 83, 66, 67, 130, 128, 146, 147)
        LONG_FMA_PAIR(84, 85, 68, 69, 130, 128, 148, 149)
        LONG_FMA_PAIR(92, 93, 76, 77, 131, 129, 148, 149)
        LONG_FMA_PAIR(116, 117, 100, 101, 134, 132, 148, 149)
        LONG_FMA_PAIR(124, 125, 108, 109, 135, 133, 148, 149)
        LONG_FMA_PAIR(126, 127, 110, 111, 135, 133, 150, 151)
        LONG_FMA_PAIR(118, 119, 102, 103, 134, 132, 150, 151)
        LONG_FMA_PAIR(94, 95, 78, 79, 131, 129, 150, 151)
        LONG_FMA_PAIR(86, 87, 70, 71, 130, 128, 150, 151)
        : LONG_ACC_ROW(0, 0), LONG_ACC_ROW(0, 1), LONG_ACC_ROW(0, 2), LONG_ACC_ROW(0, 3),
          LONG_ACC_ROW(0, 4), LONG_ACC_ROW(0, 5), LONG_ACC_ROW(0, 6), LONG_ACC_ROW(0, 7),
          LONG_ACC_ROW(1, 0), LONG_ACC_ROW(1, 1), LONG_ACC_ROW(1, 2), LONG_ACC_ROW(1, 3),
          LONG_ACC_ROW(1, 4), LONG_ACC_ROW(1, 5), LONG_ACC_ROW(1, 6), LONG_ACC_ROW(1, 7)
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]),
          "f"(b[0][0]), "f"(b[0][1]), "f"(b[0][2]), "f"(b[0][3]),
          "f"(b[0][4]), "f"(b[0][5]), "f"(b[0][6]), "f"(b[0][7]),
          "f"(b[1][0]), "f"(b[1][1]), "f"(b[1][2]), "f"(b[1][3]),
          "f"(b[1][4]), "f"(b[1][5]), "f"(b[1][6]), "f"(b[1][7]));
}

#undef LONG_ACC_ROW
#undef LONG_FMA_PAIR

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
        register_mma_8x16(c, av, bv);
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
