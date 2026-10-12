#pragma once

namespace score_bounds {

// Q/K squared norms are rounded upward at every multiply and sum. This is
// O(Sd) preparation, independent of attention's O(S^2 d) matrix products.
__global__ void prepare(
    const float *q, const float *k, const float *v, unsigned *bounds, const int elements) {
    const int e = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float qnorm = 0.0f, knorm = 0.0f, vmax = 0.0f;
    if (e < elements) {
        const float4 a = *reinterpret_cast<const float4*>(q + e);
        const float4 b = *reinterpret_cast<const float4*>(k + e);
        const float4 c = *reinterpret_cast<const float4*>(v + e);
        const float qa[4] = {a.x, a.y, a.z, a.w};
        const float ka[4] = {b.x, b.y, b.z, b.w};
        const float va[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qnorm = __fadd_ru(qnorm, __fmul_ru(qa[i], qa[i]));
            knorm = __fadd_ru(knorm, __fmul_ru(ka[i], ka[i]));
            vmax = isfinite(va[i]) ? fmaxf(vmax, fabsf(va[i])) : CUDART_INF_F;
        }
    }
#pragma unroll
    for (int distance = 8; distance; distance /= 2) {
        qnorm = __fadd_ru(qnorm, __shfl_xor_sync(0xffffffffu, qnorm, distance, 16));
        knorm = __fadd_ru(knorm, __shfl_xor_sync(0xffffffffu, knorm, distance, 16));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, distance, 16));
    }
    if (threadIdx.x % 16 == 0) {
        const float qbound = isfinite(qnorm) ? __fsqrt_ru(qnorm) : CUDART_INF_F;
        const float kbound = isfinite(knorm) ? __fsqrt_ru(knorm) : CUDART_INF_F;
        atomicMax(bounds, __float_as_uint(qbound));
        atomicMax(bounds + 1, __float_as_uint(kbound));
        atomicMax(bounds + 2, __float_as_uint(vmax));
    }
}

__device__ __forceinline__ float base(const float *bounds, const float scale) {
    // Margin covers the 64 ordered FMA dot-product roundings. The resulting
    // integer base makes score*scale + (64-base) a single FP32 FMA.
    return ceilf(__fmul_ru(__fmul_ru(bounds[0], bounds[1]),
                          __fmul_ru(scale, 1.0001f)));
}

__device__ __forceinline__ bool eligible(const float *bounds, const float scale) {
    const float maximum = base(bounds, scale);
    return isfinite(maximum) && maximum <= 32.0f && bounds[2] <= 1.0e20f;
}

} // namespace score_bounds
