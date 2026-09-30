#include "fa/attention.h"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_common.cuh"

namespace fa {
namespace {

constexpr int kElementBlockX = 16;
constexpr int kElementBlockY = 16;
constexpr int kReductionThreads = 256;

struct NaiveForwardWorkspace {
    float* scores;
    float* probabilities;
};

struct NaiveBackwardWorkspace {
    float* scores;
    float* probabilities;
    float* grad_probabilities;
    float* grad_scores;
    float* correction;
};

NaiveForwardWorkspace split_forward_workspace(void* workspace, const AttentionShape& shape) {
    auto* base = static_cast<float*>(workspace);
    const std::size_t score_elements = shape.score_elements();
    return {base, base + score_elements};
}

NaiveBackwardWorkspace split_backward_workspace(void* workspace, const AttentionShape& shape) {
    auto* base = static_cast<float*>(workspace);
    const std::size_t score_elements = shape.score_elements();
    return {
        base,
        base + score_elements,
        base + 2 * score_elements,
        base + 3 * score_elements,
        base + 4 * score_elements,
    };
}

__global__ void scores_kernel(
    const float* query,
    const float* key,
    float* scores,
    int query_count,
    int key_count,
    int head_dim,
    float scale,
    bool causal) {
    const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || key_index >= key_count) {
        return;
    }

    if (causal && key_index > query_index) {
        scores[static_cast<std::size_t>(query_index) * key_count + key_index] = -CUDART_INF_F;
        return;
    }

    float score = 0.0F;
    for (int feature = 0; feature < head_dim; ++feature) {
        score += query[static_cast<std::size_t>(query_index) * head_dim + feature] *
                 key[static_cast<std::size_t>(key_index) * head_dim + feature];
    }
    scores[static_cast<std::size_t>(query_index) * key_count + key_index] = score * scale;
}

__global__ void softmax_rows_kernel(
    const float* scores,
    float* probabilities,
    float* logsumexp,
    int query_count,
    int key_count) {
    extern __shared__ float reduction[];
    const int query_index = blockIdx.x;
    const int thread_index = threadIdx.x;
    if (query_index >= query_count) {
        return;
    }

    float local_max = -CUDART_INF_F;
    for (int key_index = thread_index; key_index < key_count; key_index += blockDim.x) {
        local_max = fmaxf(local_max, scores[static_cast<std::size_t>(query_index) * key_count + key_index]);
    }
    reduction[thread_index] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread_index < offset) {
            reduction[thread_index] = fmaxf(reduction[thread_index], reduction[thread_index + offset]);
        }
        __syncthreads();
    }
    const float row_max = reduction[0];

    float local_sum = 0.0F;
    for (int key_index = thread_index; key_index < key_count; key_index += blockDim.x) {
        local_sum += expf(scores[static_cast<std::size_t>(query_index) * key_count + key_index] - row_max);
    }
    reduction[thread_index] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread_index < offset) {
            reduction[thread_index] += reduction[thread_index + offset];
        }
        __syncthreads();
    }
    const float denominator = reduction[0];

    for (int key_index = thread_index; key_index < key_count; key_index += blockDim.x) {
        probabilities[static_cast<std::size_t>(query_index) * key_count + key_index] =
            expf(scores[static_cast<std::size_t>(query_index) * key_count + key_index] - row_max) / denominator;
    }
    if (thread_index == 0) {
        logsumexp[query_index] = row_max + logf(denominator);
    }
}

__global__ void probabilities_from_lse_kernel(
    const float* scores,
    const float* logsumexp,
    float* probabilities,
    int query_count,
    int key_count) {
    const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || key_index >= key_count) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(query_index) * key_count + key_index;
    probabilities[index] = expf(scores[index] - logsumexp[query_index]);
}

__global__ void output_kernel(
    const float* probabilities,
    const float* value,
    float* output,
    int query_count,
    int key_count,
    int head_dim) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || feature >= head_dim) {
        return;
    }

    float accumulator = 0.0F;
    for (int key_index = 0; key_index < key_count; ++key_index) {
        accumulator += probabilities[static_cast<std::size_t>(query_index) * key_count + key_index] *
                       value[static_cast<std::size_t>(key_index) * head_dim + feature];
    }
    output[static_cast<std::size_t>(query_index) * head_dim + feature] = accumulator;
}

__global__ void grad_probabilities_kernel(
    const float* grad_output,
    const float* value,
    float* grad_probabilities,
    int query_count,
    int key_count,
    int head_dim) {
    const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || key_index >= key_count) {
        return;
    }

    float accumulator = 0.0F;
    for (int feature = 0; feature < head_dim; ++feature) {
        accumulator += grad_output[static_cast<std::size_t>(query_index) * head_dim + feature] *
                       value[static_cast<std::size_t>(key_index) * head_dim + feature];
    }
    grad_probabilities[static_cast<std::size_t>(query_index) * key_count + key_index] = accumulator;
}

__global__ void correction_kernel(
    const float* output,
    const float* grad_output,
    float* correction,
    int query_count,
    int head_dim) {
    extern __shared__ float reduction[];
    const int query_index = blockIdx.x;
    const int thread_index = threadIdx.x;
    if (query_index >= query_count) {
        return;
    }

    float local_sum = 0.0F;
    for (int feature = thread_index; feature < head_dim; feature += blockDim.x) {
        local_sum += output[static_cast<std::size_t>(query_index) * head_dim + feature] *
                     grad_output[static_cast<std::size_t>(query_index) * head_dim + feature];
    }
    reduction[thread_index] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread_index < offset) {
            reduction[thread_index] += reduction[thread_index + offset];
        }
        __syncthreads();
    }
    if (thread_index == 0) {
        correction[query_index] = reduction[0];
    }
}

__global__ void grad_scores_kernel(
    const float* probabilities,
    const float* grad_probabilities,
    const float* correction,
    float* grad_scores,
    int query_count,
    int key_count) {
    const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || key_index >= key_count) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(query_index) * key_count + key_index;
    grad_scores[index] = probabilities[index] * (grad_probabilities[index] - correction[query_index]);
}

__global__ void grad_query_kernel(
    const float* grad_scores,
    const float* key,
    float* grad_query,
    int query_count,
    int key_count,
    int head_dim,
    float scale) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_index >= query_count || feature >= head_dim) {
        return;
    }
    float accumulator = 0.0F;
    for (int key_index = 0; key_index < key_count; ++key_index) {
        accumulator += grad_scores[static_cast<std::size_t>(query_index) * key_count + key_index] *
                       key[static_cast<std::size_t>(key_index) * head_dim + feature];
    }
    grad_query[static_cast<std::size_t>(query_index) * head_dim + feature] = accumulator * scale;
}

__global__ void grad_key_value_kernel(
    const float* grad_scores,
    const float* probabilities,
    const float* query,
    const float* grad_output,
    float* grad_key,
    float* grad_value,
    int query_count,
    int key_count,
    int head_dim,
    float scale) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    const int key_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (key_index >= key_count || feature >= head_dim) {
        return;
    }

    float key_accumulator = 0.0F;
    float value_accumulator = 0.0F;
    for (int query_index = 0; query_index < query_count; ++query_index) {
        key_accumulator += grad_scores[static_cast<std::size_t>(query_index) * key_count + key_index] *
                           query[static_cast<std::size_t>(query_index) * head_dim + feature];
        value_accumulator += probabilities[static_cast<std::size_t>(query_index) * key_count + key_index] *
                             grad_output[static_cast<std::size_t>(query_index) * head_dim + feature];
    }
    grad_key[static_cast<std::size_t>(key_index) * head_dim + feature] = key_accumulator * scale;
    grad_value[static_cast<std::size_t>(key_index) * head_dim + feature] = value_accumulator;
}

dim3 score_grid(const AttentionShape& shape) {
    return {
        static_cast<unsigned>(detail::ceil_div(shape.key_count, kElementBlockX)),
        static_cast<unsigned>(detail::ceil_div(shape.query_count, kElementBlockY)),
    };
}

dim3 query_feature_grid(const AttentionShape& shape) {
    return {
        static_cast<unsigned>(detail::ceil_div(shape.head_dim, kElementBlockX)),
        static_cast<unsigned>(detail::ceil_div(shape.query_count, kElementBlockY)),
    };
}

dim3 key_feature_grid(const AttentionShape& shape) {
    return {
        static_cast<unsigned>(detail::ceil_div(shape.head_dim, kElementBlockX)),
        static_cast<unsigned>(detail::ceil_div(shape.key_count, kElementBlockY)),
    };
}

}  // namespace

std::size_t naive_forward_workspace_bytes(const AttentionShape& shape) {
    return 2 * shape.score_elements() * sizeof(float);
}

std::size_t naive_backward_workspace_bytes(const AttentionShape& shape) {
    return (4 * shape.score_elements() + static_cast<std::size_t>(shape.query_count)) * sizeof(float);
}

void naive_forward(const ForwardParams& params) {
    detail::validate_forward_params(params);
    const std::size_t required_bytes = naive_forward_workspace_bytes(params.shape);
    detail::require_workspace(params.workspace, params.workspace_bytes, required_bytes);
    const auto workspace = split_forward_workspace(params.workspace, params.shape);
    const float scale = 1.0F / std::sqrt(static_cast<float>(params.shape.head_dim));
    const dim3 element_block(kElementBlockX, kElementBlockY);

    scores_kernel<<<score_grid(params.shape), element_block, 0, params.stream>>>(
        params.query,
        params.key,
        workspace.scores,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale,
        params.shape.causal);
    softmax_rows_kernel<<<params.shape.query_count, kReductionThreads, kReductionThreads * sizeof(float), params.stream>>>(
        workspace.scores,
        workspace.probabilities,
        params.logsumexp,
        params.shape.query_count,
        params.shape.key_count);
    output_kernel<<<query_feature_grid(params.shape), element_block, 0, params.stream>>>(
        workspace.probabilities,
        params.value,
        params.output,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim);
    FA_CUDA_CHECK(cudaGetLastError());
}

void naive_backward(const BackwardParams& params) {
    detail::validate_backward_params(params);
    const std::size_t required_bytes = naive_backward_workspace_bytes(params.shape);
    detail::require_workspace(params.workspace, params.workspace_bytes, required_bytes);
    const auto workspace = split_backward_workspace(params.workspace, params.shape);
    const float scale = 1.0F / std::sqrt(static_cast<float>(params.shape.head_dim));
    const dim3 element_block(kElementBlockX, kElementBlockY);

    scores_kernel<<<score_grid(params.shape), element_block, 0, params.stream>>>(
        params.query,
        params.key,
        workspace.scores,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale,
        params.shape.causal);
    probabilities_from_lse_kernel<<<score_grid(params.shape), element_block, 0, params.stream>>>(
        workspace.scores,
        params.logsumexp,
        workspace.probabilities,
        params.shape.query_count,
        params.shape.key_count);
    grad_probabilities_kernel<<<score_grid(params.shape), element_block, 0, params.stream>>>(
        params.grad_output,
        params.value,
        workspace.grad_probabilities,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim);
    correction_kernel<<<params.shape.query_count, kReductionThreads, kReductionThreads * sizeof(float), params.stream>>>(
        params.output,
        params.grad_output,
        workspace.correction,
        params.shape.query_count,
        params.shape.head_dim);
    grad_scores_kernel<<<score_grid(params.shape), element_block, 0, params.stream>>>(
        workspace.probabilities,
        workspace.grad_probabilities,
        workspace.correction,
        workspace.grad_scores,
        params.shape.query_count,
        params.shape.key_count);
    grad_query_kernel<<<query_feature_grid(params.shape), element_block, 0, params.stream>>>(
        workspace.grad_scores,
        params.key,
        params.grad_query,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale);
    grad_key_value_kernel<<<key_feature_grid(params.shape), element_block, 0, params.stream>>>(
        workspace.grad_scores,
        workspace.probabilities,
        params.query,
        params.grad_output,
        params.grad_key,
        params.grad_value,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale);
    FA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fa
