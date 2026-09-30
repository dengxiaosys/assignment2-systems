#include "fa/attention.h"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_common.cuh"

namespace fa {
namespace {

constexpr int kFeatureSlots = detail::kMaxHeadDim / detail::kWarpSize;

std::size_t forward_shared_memory_bytes(const AttentionShape& shape) {
    const std::size_t query_values = static_cast<std::size_t>(detail::kQueriesPerTile) * shape.head_dim;
    const std::size_t key_value_values = 2ULL * detail::kKeysPerTile * shape.head_dim;
    return (query_values + key_value_values) * sizeof(float);
}

// Each warp owns one query row. Its lanes stream 32 keys and own strided
// feature positions in the output row, so no output atomics are needed.
__global__ void tiled_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* logsumexp,
    int query_count,
    int key_count,
    int head_dim,
    float scale,
    bool causal) {
    extern __shared__ float shared[];
    float* shared_query = shared;
    float* shared_key = shared_query + detail::kQueriesPerTile * head_dim;
    float* shared_value = shared_key + detail::kKeysPerTile * head_dim;

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
        shared_query[index] =
            global_query < query_count ? query[static_cast<std::size_t>(global_query) * head_dim + feature] : 0.0F;
    }
    __syncthreads();

    float output_accumulators[kFeatureSlots] = {};
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0F;

    for (int key_start = 0; key_start < key_count; key_start += detail::kKeysPerTile) {
        for (int index = thread_index; index < detail::kKeysPerTile * head_dim; index += blockDim.x) {
            const int local_key = index / head_dim;
            const int feature = index % head_dim;
            const int global_key = key_start + local_key;
            const float loaded_key =
                global_key < key_count ? key[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
            const float loaded_value =
                global_key < key_count ? value[static_cast<std::size_t>(global_key) * head_dim + feature] : 0.0F;
            shared_key[index] = loaded_key;
            shared_value[index] = loaded_value;
        }
        __syncthreads();

        const bool tile_has_visible_key = valid_query && key_start < key_count && (!causal || key_start <= query_index);
        if (tile_has_visible_key) {
            const int key_index = key_start + lane_index;
            const bool visible = key_index < key_count && (!causal || key_index <= query_index);
            float score = -CUDART_INF_F;
            if (visible) {
                score = 0.0F;
                for (int feature = 0; feature < head_dim; ++feature) {
                    score += shared_query[warp_index * head_dim + feature] *
                             shared_key[lane_index * head_dim + feature];
                }
                score *= scale;
            }

            const float tile_max = detail::warp_allreduce_max(score);
            const float next_max = fmaxf(running_max, tile_max);
            const float old_scale = isinf(running_max) ? 0.0F : expf(running_max - next_max);
            const float probability_numerator = visible ? expf(score - next_max) : 0.0F;
            const float tile_sum = detail::warp_allreduce_sum(probability_numerator);

#pragma unroll
            for (int slot = 0; slot < kFeatureSlots; ++slot) {
                const int feature = lane_index + slot * detail::kWarpSize;
                float tile_output = 0.0F;
#pragma unroll
                for (int source_lane = 0; source_lane < detail::kWarpSize; ++source_lane) {
                    const float source_probability =
                        __shfl_sync(detail::kFullWarpMask, probability_numerator, source_lane);
                    if (feature < head_dim) {
                        tile_output += source_probability * shared_value[source_lane * head_dim + feature];
                    }
                }
                if (feature < head_dim) {
                    output_accumulators[slot] = old_scale * output_accumulators[slot] + tile_output;
                }
            }
            running_sum = old_scale * running_sum + tile_sum;
            running_max = next_max;
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
            output[static_cast<std::size_t>(query_index) * head_dim + feature] =
                output_accumulators[slot] / running_sum;
        }
    }
    if (lane_index == 0) {
        logsumexp[query_index] = running_max + logf(running_sum);
    }
}

}  // namespace

std::size_t tiled_forward_workspace_bytes(const AttentionShape& shape) {
    detail::validate_tiled_shape(shape);
    return 0;
}

void tiled_forward(const ForwardParams& params) {
    detail::validate_forward_params(params);
    detail::validate_tiled_shape(params.shape);
    detail::require_workspace(params.workspace, params.workspace_bytes, tiled_forward_workspace_bytes(params.shape));

    const std::size_t shared_bytes = forward_shared_memory_bytes(params.shape);
    detail::validate_dynamic_shared_memory(shared_bytes);
    const int block_count = detail::ceil_div(params.shape.query_count, detail::kQueriesPerTile);
    const float scale = 1.0F / std::sqrt(static_cast<float>(params.shape.head_dim));
    tiled_forward_kernel<<<block_count, detail::kThreadsPerBlock, shared_bytes, params.stream>>>(
        params.query,
        params.key,
        params.value,
        params.output,
        params.logsumexp,
        params.shape.query_count,
        params.shape.key_count,
        params.shape.head_dim,
        scale,
        params.shape.causal);
    FA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fa
