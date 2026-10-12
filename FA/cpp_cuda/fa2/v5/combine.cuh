#pragma once

#include "score_bounds.cuh"

namespace split_attention {

// Every split uses the same exponent base, so the unnormalized states add
// directly. Empty causal splits write zeros and are normalized only here.
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

} // namespace split_attention
