#pragma once

namespace persistent_q {

constexpr unsigned MASK = 0xffffffffu;

__device__ __forceinline__ float exp2(const float x) {
    float value;
    asm("ex2.approx.f32 %0, %1;" : "=f"(value) : "f"(x));
    return value;
}

__device__ __forceinline__ int p_row(const int query, const int key) {
    return query ^ (((key / 4) % 8) * 4);
}

__device__ __forceinline__ void transpose(float (&out)[64][68], const float *in) {
#pragma unroll 1
    for (int group = 0; group < 8; ++group) {
        const int e = threadIdx.x * 4 + group * 512;
        const float4 a = *reinterpret_cast<const float4*>(in + e);
        const int row = e / 64;
        const int dim = e % 64;
        out[dim][row] = a.x;
        out[dim + 1][row] = a.y;
        out[dim + 2][row] = a.z;
        out[dim + 3][row] = a.w;
    }
}

__device__ __forceinline__ void load_v(float (&out)[4], const float *in) {
    const float4 a = *reinterpret_cast<const float4*>(in + threadIdx.x * 4);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
}

__device__ __forceinline__ void store_v(float (&out)[8][64], const float (&in)[4]) {
    const int e = threadIdx.x * 4;
    *reinterpret_cast<float4*>(&out[e / 64][e % 64]) =
        make_float4(in[0], in[1], in[2], in[3]);
}

__device__ __forceinline__ void outer(
    float (&c)[4][8], const float (&a)[4], const float (&b)[8]) {
#pragma unroll
    for (int col = 0; col < 8; ++col) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const int r = col % 2 ? 3 - row : row;
            c[r][col] += a[r] * b[col];
        }
    }
}

template <bool CAUSAL>
__global__ void forward(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ o, const int seq_len, const float scale) {
    struct alignas(16) Storage {
        float q[64][68];
        union {
            float k[64][68];
            struct {
                float p[64][68];
                float v[2][8][64];
            } pv;
        };
    };
    __shared__ Storage shared;
    const int query = (threadIdx.x / 32) * 16 + (threadIdx.x % 32 / 8) * 4;
    const int column = (threadIdx.x % 8) * 4;
    const int query_base = blockIdx.x * 64;
    const int key_end = CAUSAL ? query_base + 64 : seq_len;
    float output[4][8] = {};
    float maximum[4], denominator[4] = {};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        maximum[r] = -CUDART_INF_F;
    }
    transpose(shared.q, q + query_base * 64);
    __syncthreads();

    for (int tile = 0; tile < key_end; tile += 64) {
        transpose(shared.k, k + tile * 64);
        __syncthreads();
        float score[4][8] = {};
#pragma unroll 4
        for (int d = 0; d < 64; ++d) {
            float a[4], b[8];
            const float4 aq = *reinterpret_cast<const float4*>(&shared.q[d][query]);
            a[0] = aq.x;
            a[1] = aq.y;
            a[2] = aq.z;
            a[3] = aq.w;
            const float4 bk = *reinterpret_cast<const float4*>(&shared.k[d][column]);
            const float4 bh = *reinterpret_cast<const float4*>(&shared.k[d][column + 32]);
            b[0] = bk.x;
            b[1] = bk.y;
            b[2] = bk.z;
            b[3] = bk.w;
            b[4] = bh.x;
            b[5] = bh.y;
            b[6] = bh.z;
            b[7] = bh.w;
            outer(score, a, b);
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                score[r][c] *= scale;
                if constexpr (CAUSAL) {
                    if (tile == query_base && column + (c / 4) * 32 + c % 4 > query + r) {
                        score[r][c] = -CUDART_INF_F;
                    }
                }
                m = fmaxf(m, score[r][c]);
            }
#pragma unroll
            for (int stride = 4; stride; stride /= 2) {
                m = fmaxf(m, __shfl_xor_sync(MASK, m, stride, 8));
            }
            m = fmaxf(m, maximum[r]);
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
            const int key = column + (c / 4) * 32 + c % 4;
            *reinterpret_cast<float4*>(&shared.pv.p[key][p_row(query, key)]) =
                make_float4(score[0][c], score[1][c], score[2][c], score[3][c]);
        }
        float vg[4];
        load_v(vg, v + tile * 64);
        store_v(shared.pv.v[0], vg);
        __syncthreads();
#pragma unroll 1
        for (int stage = 0; stage < 64; stage += 8) {
            const int read = (stage / 8) % 2;
            if (stage + 8 < 64) {
                load_v(vg, v + (tile + stage + 8) * 64);
            }
#pragma unroll 4
            for (int d = 0; d < 8; ++d) {
                float a[4], b[8];
                const float4 ap = *reinterpret_cast<const float4*>(
                    &shared.pv.p[stage + d][p_row(query, stage + d)]);
                a[0] = ap.x;
                a[1] = ap.y;
                a[2] = ap.z;
                a[3] = ap.w;
                const float4 bv = *reinterpret_cast<const float4*>(&shared.pv.v[read][d][column]);
                const float4 bh = *reinterpret_cast<const float4*>(&shared.pv.v[read][d][column + 32]);
                b[0] = bv.x;
                b[1] = bv.y;
                b[2] = bv.z;
                b[3] = bv.w;
                b[4] = bh.x;
                b[5] = bh.y;
                b[6] = bh.z;
                b[7] = bh.w;
                outer(output, a, b);
            }
            if (stage + 8 < 64) {
                store_v(shared.pv.v[read ^ 1], vg);
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        float total = denominator[r];
#pragma unroll
        for (int stride = 4; stride; stride /= 2) {
            total += __shfl_xor_sync(MASK, total, stride, 8);
        }
        const float inv = 1.0f / total;
#pragma unroll
        for (int c = 0; c < 8; c += 4) {
            *reinterpret_cast<float4*>(&o[(query_base + query + r) * 64 + column + (c / 4) * 32]) =
                make_float4(output[r][c] * inv, output[r][c + 1] * inv,
                            output[r][c + 2] * inv, output[r][c + 3] * inv);
        }
    }
}

} // namespace persistent_q
