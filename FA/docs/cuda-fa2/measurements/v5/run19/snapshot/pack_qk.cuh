#pragma once

namespace packed_qk {

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
