#pragma once

namespace packed_qk {

// Fuse the conservative score bound into the existing bit-preserving pack.
__global__ void prepare_with_bounds(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ packed, unsigned *bounds) {
    __shared__ float tile[32][33];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.x * 64 + blockIdx.y * 32;
    const bool query = blockIdx.z == 0;
    const float *input = query ? q : k;
    float *output = packed + (query ? 0 : gridDim.x * 64 * 64);
    float squared[4] = {}, vmax[4] = {};

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int dimension = half * 32;
#pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int e = (row + y + j) * 64 + dimension + x;
            const float value = input[e];
            tile[y + j][x] = value;
            squared[j / 8] = __fadd_ru(squared[j / 8], __fmul_ru(value, value));
            if (query) {
                const float a = v[e];
                vmax[j / 8] = isfinite(a) ? fmaxf(vmax[j / 8], fabsf(a)) : CUDART_INF_F;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 32; j += 8) {
            output[blockIdx.x * 4096 + (dimension + y + j) * 64 + blockIdx.y * 32 + x] =
                tile[x][y + j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int distance = 16; distance; distance /= 2) {
            squared[j] = __fadd_ru(squared[j], __shfl_xor_sync(0xffffffffu, squared[j], distance));
            vmax[j] = fmaxf(vmax[j], __shfl_xor_sync(0xffffffffu, vmax[j], distance));
        }
        if (x == 0) {
            const float norm = isfinite(squared[j]) ? __fsqrt_ru(squared[j]) : CUDART_INF_F;
            atomicMax(bounds + (query ? 0 : 1), __float_as_uint(norm));
            if (query) {
                atomicMax(bounds + 2, __float_as_uint(vmax[j]));
            }
        }
    }
}

// Each CTA transposes one 64-row, d=64 tile without changing any FP32 bits.
__global__ void prepare(
    const float *__restrict__ q, const float *__restrict__ k,
    float *__restrict__ packed) {
    __shared__ float tile[32][33];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.x * 64 + blockIdx.y * 32;
    const int dimension = (blockIdx.z % 2) * 32;
    const float *input = blockIdx.z < 2 ? q : k;
    float *output = packed + (blockIdx.z < 2 ? 0 : gridDim.x * 64 * 64);

#pragma unroll
    for (int j = 0; j < 32; j += 8) {
        tile[y + j][x] = input[(row + y + j) * 64 + dimension + x];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
        output[blockIdx.x * 4096 + (dimension + y + j) * 64 + blockIdx.y * 32 + x] =
            tile[x][y + j];
    }
}

} // namespace packed_qk
