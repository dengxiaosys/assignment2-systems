#include "fa/attention.h"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "cuda_common.cuh"

namespace fa {
namespace {

constexpr int kFeatureSlots = detail::kMaxHeadDim / detail::kWarpSize;
constexpr int kOwnedRowsPerBlock = detail::kWarpsPerBlock;
constexpr int kStreamedRowsPerTile = detail::kWarpSize;

std::size_t backward_shared_memory_bytes(const AttentionShape& shape) {
    const int shared_stride = detail::shared_row_stride(shape.head_dim);
    const std::size_t owned_values = 2ULL * kOwnedRowsPerBlock * shared_stride;
    const std::size_t streamed_values = 2ULL * kStreamedRowsPerTile * shared_stride;
    return (owned_values + streamed_values) * sizeof(float);
}

// Computes D_i = dot(O_i, dO_i), the row correction reused by both passes.
__global__ void correction_kernel(
    const float* output,
    const float* grad_output,
    float* correction,
    int query_count,
    int head_dim) {
    const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= query_count) {
        return;
    }
    float accumulator = 0.0F;
    for (int feature = 0; feature < head_dim; ++feature) {
        accumulator += output[static_cast<std::size_t>(query_index) * head_dim + feature] *
                       grad_output[static_cast<std::size_t>(query_index) * head_dim + feature];
    }
    correction[query_index] = accumulator;
}

// Each warp owns one query row and scans all key tiles, fully reducing dQ
// before writing it once.
__global__ void tiled_grad_query_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* logsumexp,
    const float* grad_output,
    const float* correction,
    float* grad_query,
    int query_count,
    int key_count,
    int head_dim,
    float scale,
    bool causal) {
    extern __shared__ float shared[];
    const int shared_stride = detail::shared_row_stride(head_dim);
    float* shared_query = shared;
    float* shared_grad_output = shared_query + detail::kQueriesPerTile * shared_stride;
    float* shared_key = shared_grad_output + detail::kQueriesPerTile * shared_stride;
    float* shared_value = shared_key + detail::kKeysPerTile * shared_stride;

    const int thread_index = threadIdx.x;
    const int warp_index = thread_index / detail::kWarpSize;
    const int lane_index = thread_index % detail::kWarpSize;
    const int query_start = blockIdx.x * detail::kQueriesPerTile;
    const int query_index = query_start + warp_index;
    const bool valid_query = query_index < query_count;

    for (int index = thread_index; index < detail::kQueriesPerTile * head_dim; index += blockDim.x) {
        const int local_query = index / head_dim;
        const int feature = index % head_dim;
        const int global_query = query_start + local_query;
        const int shared_index = local_query * shared_stride + feature;
        shared_query[shared_index] =
            global_query < query_count ? query[static_cast<std::size_t>(global_query) * head_dim + feature] : 0.0F;
        shared_grad_output[shared_index] =
            global_query < query_count
                ? grad_output[static_cast<std::size_t>(global_query) * head_dim + feature]
                : 0.0F;
    }
    __syncthreads();

    float grad_query_accumulators[kFeatureSlots] = {};
    const float row_logsumexp = valid_query ? logsumexp[query_index] : 0.0F;
    const float row_correction = valid_query ? correction[query_index] : 0.0F;

    for (int key_start = 0; key_start < key_count; key_start += detail::kKeysPerTile) {
        for (int index = thread_index; index < detail::kKeysPerTile * head_dim; index += blockDim.x) {
            const int local_key = index / head_dim;
            const int feature = index % head_dim;
            const int global_key = key_start + local_key;
            const int shared_index = local_key * shared_stride + feature;
            shared_key[shared_index] =
                global_key < key_count ? key[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
            shared_value[shared_index] =
                global_key < key_count ? value[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
        }
        __syncthreads();

        const bool tile_has_visible_key = valid_query && key_start < key_count && (!causal || key_start <= query_index);
        if (tile_has_visible_key) {
            const int key_index = key_start + lane_index;
            const bool visible = key_index < key_count && (!causal || key_index <= query_index);
            float probability = 0.0F;
            float grad_score = 0.0F;
            if (visible) {
                float score = 0.0F;
                float grad_probability = 0.0F;
                for (int feature = 0; feature < head_dim; ++feature) {
                    score += shared_query[warp_index * shared_stride + feature] *
                             shared_key[lane_index * shared_stride + feature];
                    grad_probability += shared_grad_output[warp_index * shared_stride + feature] *
                                        shared_value[lane_index * shared_stride + feature];
                }
                score *= scale;
                probability = expf(score - row_logsumexp);
                grad_score = probability * (grad_probability - row_correction);
            }

#pragma unroll
            for (int slot = 0; slot < kFeatureSlots; ++slot) {
                const int feature_base = slot * detail::kWarpSize;
                if (feature_base < head_dim) {
                    const int feature = lane_index + feature_base;
                    float tile_contribution = 0.0F;
#pragma unroll
                    for (int source_lane = 0; source_lane < detail::kWarpSize; ++source_lane) {
                        const float source_grad_score =
                            __shfl_sync(detail::kFullWarpMask, grad_score, source_lane);
                        if (feature < head_dim) {
                            tile_contribution +=
                                source_grad_score * shared_key[source_lane * shared_stride + feature];
                        }
                    }
                    if (feature < head_dim) {
                        grad_query_accumulators[slot] += tile_contribution * scale;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!valid_query) {
        return;
    }
#pragma unroll
    for (int slot = 0; slot < kFeatureSlots; ++slot) {
        const int feature = lane_index + slot * detail::kWarpSize;
        if (feature < head_dim) {
            grad_query[static_cast<std::size_t>(query_index) * head_dim + feature] =
                grad_query_accumulators[slot];
        }
    }
}

// Each warp owns one key row and scans all query tiles, fully reducing dK and
// dV before writing them once.
__global__ void tiled_grad_key_value_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* logsumexp,
    const float* grad_output,
    const float* correction,
    float* grad_key,
    float* grad_value,
    int query_count,
    int key_count,
    int head_dim,
    float scale,
    bool causal) {
    extern __shared__ float shared[];
    const int shared_stride = detail::shared_row_stride(head_dim);
    float* shared_key = shared;
    float* shared_value = shared_key + kOwnedRowsPerBlock * shared_stride;
    float* shared_query = shared_value + kOwnedRowsPerBlock * shared_stride;
    float* shared_grad_output = shared_query + kStreamedRowsPerTile * shared_stride;

    const int thread_index = threadIdx.x;
    const int warp_index = thread_index / detail::kWarpSize;
    const int lane_index = thread_index % detail::kWarpSize;
    const int key_start = blockIdx.x * kOwnedRowsPerBlock;
    const int key_index = key_start + warp_index;
    const bool valid_key = key_index < key_count;

    for (int index = thread_index; index < kOwnedRowsPerBlock * head_dim; index += blockDim.x) {
        const int local_key = index / head_dim;
        const int feature = index % head_dim;
        const int global_key = key_start + local_key;
        const int shared_index = local_key * shared_stride + feature;
        shared_key[shared_index] =
            global_key < key_count ? key[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
        shared_value[shared_index] =
            global_key < key_count ? value[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
    }
    __syncthreads();

    float grad_key_accumulators[kFeatureSlots] = {};
    float grad_value_accumulators[kFeatureSlots] = {};

    for (int query_start = 0; query_start < query_count; query_start += kStreamedRowsPerTile) {
        for (int index = thread_index; index < kStreamedRowsPerTile * head_dim; index += blockDim.x) {
            const int local_query = index / head_dim;
            const int feature = index % head_dim;
            const int global_query = query_start + local_query;
            const int shared_index = local_query * shared_stride + feature;
            shared_query[shared_index] =
                global_query < query_count ? query[static_cast<std::size_t>(global_query) * head_dim + feature] : 0.0F;
            shared_grad_output[shared_index] =
                global_query < query_count
                    ? grad_output[static_cast<std::size_t>(global_query) * head_dim + feature]
                    : 0.0F;
        }
        __syncthreads();

        if (valid_key) {
            const int query_index = query_start + lane_index;
            const bool visible = query_index < query_count && (!causal || key_index <= query_index);
            float probability = 0.0F;
            float grad_score = 0.0F;
            if (visible) {
                float score = 0.0F;
                float grad_probability = 0.0F;
                for (int feature = 0; feature < head_dim; ++feature) {
                    score += shared_query[lane_index * shared_stride + feature] *
                             shared_key[warp_index * shared_stride + feature];
                    grad_probability += shared_grad_output[lane_index * shared_stride + feature] *
                                        shared_value[warp_index * shared_stride + feature];
                }
                score *= scale;
                probability = expf(score - logsumexp[query_index]);
                grad_score = probability * (grad_probability - correction[query_index]);
            }

#pragma unroll
            for (int slot = 0; slot < kFeatureSlots; ++slot) {
                const int feature_base = slot * detail::kWarpSize;
                if (feature_base < head_dim) {
                    const int feature = lane_index + feature_base;
                    float key_contribution = 0.0F;
                    float value_contribution = 0.0F;
#pragma unroll
                    for (int source_lane = 0; source_lane < detail::kWarpSize; ++source_lane) {
                        const float source_grad_score =
                            __shfl_sync(detail::kFullWarpMask, grad_score, source_lane);
                        const float source_probability =
                            __shfl_sync(detail::kFullWarpMask, probability, source_lane);
                        if (feature < head_dim) {
                            key_contribution +=
                                source_grad_score * shared_query[source_lane * shared_stride + feature];
                            value_contribution +=
                                source_probability * shared_grad_output[source_lane * shared_stride + feature];
                        }
                    }
                    if (feature < head_dim) {
                        grad_key_accumulators[slot] += key_contribution * scale;
                        grad_value_accumulators[slot] += value_contribution;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!valid_key) {
        return;
    }
#pragma unroll
    for (int slot = 0; slot < kFeatureSlots; ++slot) {
        const int feature = lane_index + slot * detail::kWarpSize;
        if (feature < head_dim) {
            const std::size_t index = static_cast<std::size_t>(key_index) * head_dim + feature;
            grad_key[index] = grad_key_accumulators[slot];
            grad_value[index] = grad_value_accumulators[slot];
        }
    }
}

}  // namespace

std::size_t tiled_backward_workspace_bytes(const AttentionShape& shape) {
    detail::validate_tiled_shape(shape);
    return static_cast<std::size_t>(shape.query_count) * sizeof(float);
}

void tiled_backward(const BackwardParams& params) {
    detail::validate_backward_params(params);
    detail::validate_tiled_shape(params.shape);
    const std::size_t required_bytes = tiled_backward_workspace_bytes(params.shape);
    detail::require_workspace(params.workspace, params.workspace_bytes, required_bytes);
    auto* correction = static_cast<float*>(params.workspace);

    constexpr int correction_threads = 128;
    const int correction_blocks = detail::ceil_div(params.shape.query_count, correction_threads);
    correction_kernel<<<correction_blocks, correction_threads, 0, params.stream>>>(
        params.output,
        params.grad_output,
        correction,
        params.shape.query_count,
        params.shape.head_dim);

    const std::size_t shared_bytes = backward_shared_memory_bytes(params.shape);
    detail::validate_dynamic_shared_memory(shared_bytes);
    const float scale = 1.0F / std::sqrt(static_cast<float>(params.shape.head_dim));

    const int key_blocks = detail::ceil_div(params.shape.key_count, kOwnedRowsPerBlock);
    tiled_grad_key_value_kernel<<<key_blocks, detail::kThreadsPerBlock, shared_bytes, params.stream>>>(
        params.query,
        params.key,
        params.value,
        params.logsumexp,
        params.grad_output,
        correction,
        params.grad_key,
        params.grad_value,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale,
        params.shape.causal);

    const int query_blocks = detail::ceil_div(params.shape.query_count, detail::kQueriesPerTile);
    tiled_grad_query_kernel<<<query_blocks, detail::kThreadsPerBlock, shared_bytes, params.stream>>>(
        params.query,
        params.key,
        params.value,
        params.logsumexp,
        params.grad_output,
        correction,
        params.grad_query,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale,
        params.shape.causal);
    FA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fa
