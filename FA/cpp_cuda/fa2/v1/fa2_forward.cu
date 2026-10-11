#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kMaxHeadDim = 128;
constexpr unsigned kFullWarpMask = 0xffffffffu;

__device__ float warp_sum(float value) {
    for (int stride = kWarpSize / 2; stride > 0; stride /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, stride);
    }
    return value; // Only lane 0 contains the complete sum.
}

template <int kDimsPerLane>
__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                      const float *__restrict__ v, float *const __restrict__ o,
                                      const int seq_len, const int head_dim, const float scale,
                                      const bool is_causal) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int qi = blockIdx.x * kWarpsPerBlock + warp;
    if (qi >= seq_len) {
        return; // The entire warp exits; valid warps keep all 32 lanes.
    }

    float q_dim[kDimsPerLane];
    float z[kDimsPerLane];
#pragma unroll
    for (int r = 0; r < kDimsPerLane; ++r) {
        const int dim = lane + r * kWarpSize;
        const bool is_valid_dim = dim < head_dim;
        q_dim[r] = is_valid_dim ? q[qi * head_dim + dim] : 0.0f;
        z[r] = 0.0f; // This lane's components of the numerator vector.
    }

    float m = -CUDART_INF_F; // Running maximum; updated by lane 0.
    float l = 0.0f;          // Denominator ell; updated by lane 0.
    const int key_limit = is_causal ? qi + 1 : seq_len;

    for (int ki = 0; ki < key_limit; ++ki) {
        float reduction = 0.0f;
#pragma unroll
        for (int r = 0; r < kDimsPerLane; ++r) {
            const int dim = lane + r * kWarpSize;
            if (dim < head_dim) {
                reduction += q_dim[r] * k[ki * head_dim + dim];
            }
        }
        reduction = warp_sum(reduction);

        float alpha = 0.0f; // exp(m_old - m_new)
        float p_tilde = 0.0f;
        if (lane == 0) {
            const float s = reduction * scale;
            const float m_new = fmaxf(m, s);
            alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
            p_tilde = expf(s - m_new);
            l = alpha * l + p_tilde;
            m = m_new;
        }
        // Pascal requires all participating lanes at the same shuffle.
        alpha = __shfl_sync(kFullWarpMask, alpha, 0);
        p_tilde = __shfl_sync(kFullWarpMask, p_tilde, 0);

#pragma unroll
        for (int r = 0; r < kDimsPerLane; ++r) {
            const int dim = lane + r * kWarpSize;
            if (dim < head_dim) {
                z[r] = alpha * z[r] + p_tilde * v[ki * head_dim + dim];
            }
        }
    }

    const float l_final = __shfl_sync(kFullWarpMask, l, 0);
#pragma unroll
    for (int r = 0; r < kDimsPerLane; ++r) {
        const int dim = lane + r * kWarpSize;
        if (dim < head_dim) {
            o[qi * head_dim + dim] = z[r] / l_final;
        }
    }
}

void check_inputs(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v) {
    TORCH_CHECK(q.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "Q/K/V must have shape (S, d)");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Q/K/V must have the same shape");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "Q/K/V must use float32");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "Q/K/V must have the same dtype");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q/K/V must be on the same CUDA device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0, "S and d must be positive");
    TORCH_CHECK(q.size(1) <= kMaxHeadDim, "head dimension must be <= ", kMaxHeadDim);
    TORCH_CHECK(q.numel() <= std::numeric_limits<int>::max(),
                "Q/K/V are too large for 32-bit kernel indexing");
}

template <int kDimsPerLane>
void launch_forward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                    const int seq_len, const int head_dim, const float scale, const bool is_causal,
                    const dim3 grid, const dim3 block, const cudaStream_t stream) {
    fa2_forward_fp32_kernel<kDimsPerLane><<<grid, block, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), o.data_ptr<float>(),
        seq_len, head_dim, scale, is_causal);
}

} // namespace

at::Tensor fa2_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal) {
    check_inputs(q, k, v);
    const c10::cuda::CUDAGuard device_guard(q.device());

    const int seq_len = static_cast<int>(q.size(0));
    const int head_dim = static_cast<int>(q.size(1));
    auto o = at::empty_like(q);

    const dim3 grid(1 + (seq_len - 1) / kWarpsPerBlock);
    const dim3 block(kThreadsPerBlock);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (head_dim <= 32) {
        launch_forward<1>(q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream);
    } else if (head_dim <= 64) {
        launch_forward<2>(q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream);
    } else if (head_dim <= 96) {
        launch_forward<3>(q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream);
    } else {
        launch_forward<4>(q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
