#pragma once

namespace score_bounds {

__device__ __forceinline__ float base(const float *bounds, const float scale) {
    // The pack kernel rounds squared norms upward. This additional margin
    // covers 64 ordered FP32 dot-product FMA roundings.
    return ceilf(__fmul_ru(__fmul_ru(bounds[0], bounds[1]),
                          __fmul_ru(scale, 1.0001f)));
}

__device__ __forceinline__ bool eligible(const float *bounds, const float scale) {
    // Zero exponent base gives normal P in [2^-32, 2^32]. Together with the
    // d64 indexing bound, |V| <= 1e20 keeps partial numerators representable.
    const float maximum = base(bounds, scale);
    return isfinite(maximum) && maximum <= 32.0f && bounds[2] <= 1.0e20f;
}

} // namespace score_bounds
