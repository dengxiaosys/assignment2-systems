#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "fa/shape.h"

namespace fa {

// Tensor pointers refer to contiguous row-major FP32 device storage. The
// caller owns every tensor and the reusable device workspace.
struct ForwardParams {
    AttentionShape shape;
    const float* query;
    const float* key;
    const float* value;
    float* output;
    float* logsumexp;
    void* workspace;
    std::size_t workspace_bytes;
    cudaStream_t stream = nullptr;
};

// output and logsumexp must come from forward with the same Q/K/V and shape.
struct BackwardParams {
    AttentionShape shape;
    const float* query;
    const float* key;
    const float* value;
    const float* output;
    const float* logsumexp;
    const float* grad_output;
    float* grad_query;
    float* grad_key;
    float* grad_value;
    void* workspace;
    std::size_t workspace_bytes;
    cudaStream_t stream = nullptr;
};

[[nodiscard]] std::size_t naive_forward_workspace_bytes(const AttentionShape& shape);
[[nodiscard]] std::size_t naive_backward_workspace_bytes(const AttentionShape& shape);
[[nodiscard]] std::size_t tiled_forward_workspace_bytes(const AttentionShape& shape);
[[nodiscard]] std::size_t tiled_backward_workspace_bytes(const AttentionShape& shape);

void naive_forward(const ForwardParams& params);
void naive_backward(const BackwardParams& params);
void tiled_forward(const ForwardParams& params);
void tiled_backward(const BackwardParams& params);

}  // namespace fa
