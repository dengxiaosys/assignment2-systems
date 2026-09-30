#pragma once

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "fa/attention.h"

namespace fa::detail {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kKeysPerTile = 32;
constexpr int kQueriesPerTile = kWarpsPerBlock;
constexpr int kMaxHeadDim = 128;
constexpr unsigned kFullWarpMask = 0xffffffffu;

inline int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

inline void check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

#define FA_CUDA_CHECK(expression) ::fa::detail::check_cuda((expression), #expression, __FILE__, __LINE__)

inline void validate_forward_params(const ForwardParams& params) {
    params.shape.validate();
    if (params.query == nullptr || params.key == nullptr || params.value == nullptr || params.output == nullptr || params.logsumexp == nullptr) {
        throw std::invalid_argument("forward tensor pointers must not be null");
    }
}

inline void validate_backward_params(const BackwardParams& params) {
    params.shape.validate();
    if (params.query == nullptr || params.key == nullptr || params.value == nullptr || params.output == nullptr || params.logsumexp == nullptr ||
        params.grad_output == nullptr || params.grad_query == nullptr || params.grad_key == nullptr || params.grad_value == nullptr) {
        throw std::invalid_argument("backward tensor pointers must not be null");
    }
}

inline void require_workspace(void* workspace, std::size_t provided_bytes, std::size_t required_bytes) {
    if (provided_bytes < required_bytes) {
        throw std::invalid_argument("workspace is smaller than the implementation requires");
    }
    if (required_bytes != 0 && workspace == nullptr) {
        throw std::invalid_argument("workspace must not be null when workspace bytes are required");
    }
}

inline void validate_tiled_shape(const AttentionShape& shape) {
    shape.validate();
    if (shape.head_dim > kMaxHeadDim) {
        throw std::invalid_argument("tiled CUDA implementation supports head_dim <= 128");
    }
}

inline void validate_dynamic_shared_memory(std::size_t required_bytes) {
    constexpr std::size_t guaranteed_limit_bytes = 48 * 1024;
    if (required_bytes <= guaranteed_limit_bytes) {
        return;
    }
    int device = 0;
    int limit_bytes = 0;
    FA_CUDA_CHECK(cudaGetDevice(&device));
    FA_CUDA_CHECK(cudaDeviceGetAttribute(&limit_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (required_bytes > static_cast<std::size_t>(limit_bytes)) {
        throw std::invalid_argument("requested tiled kernel shared memory exceeds the device per-block limit");
    }
}

__device__ inline float warp_allreduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

__device__ inline float warp_allreduce_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

}  // namespace fa::detail
