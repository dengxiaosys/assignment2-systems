#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_HEAD_DIM = 128;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

__device__ float warp_sum(float value) {
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        value += __shfl_down_sync(FULL_WARP_MASK, value, stride);
    }
    return value; // Only lane 0 contains the complete sum.
}

template <int DIMS_PER_LANE>
__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                        const float *__restrict__ v, float *const __restrict__ o,
                                        const int seq_len, const int head_dim, const float scale,
                                        const bool is_causal) {
    const int warpi = threadIdx.x / WARP_SIZE;
    const int lanei = threadIdx.x % WARP_SIZE;
    const int qi = blockIdx.x * WARPS_PER_BLOCK + warpi;
    if (qi >= seq_len) {
        return; // The entire warp exits; valid warps keep all 32 lanes.
    }

    float q_dim[DIMS_PER_LANE];
    float z[DIMS_PER_LANE];
#pragma unroll
    for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
        const int dimi = ri * WARP_SIZE + lanei;
        q_dim[ri] = dimi < head_dim ? q[qi * head_dim + dimi] : 0.0f;
        z[ri] = 0.0f; // This lane's components of the numerator vector.
    }

    float m = -CUDART_INF_F; // Running maximum; updated by lane 0.
    float l = 0.0f;          // Denominator ell; updated by lane 0.
    const int key_limit = is_causal ? qi + 1 : seq_len;

    for (int ki = 0; ki < key_limit; ++ki) {
        float reduction = 0.0f;
#pragma unroll
        for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
            const int dimi = ri * WARP_SIZE + lanei;
            if (dimi < head_dim) {
                reduction += q_dim[ri] * k[ki * head_dim + dimi];
            }
        }
        reduction = warp_sum(reduction);

        float alpha = 0.0f; // exp(m_old - m_new)
        float p_tilde = 0.0f;
        if (lanei == 0) {
            const float s = reduction * scale;
            const float m_new = fmaxf(m, s);
            alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
            p_tilde = expf(s - m_new);
            l = alpha * l + p_tilde;
            m = m_new;
        }
        // Pascal requires all participating lanes at the same shuffle.
        alpha = __shfl_sync(FULL_WARP_MASK, alpha, 0);
        p_tilde = __shfl_sync(FULL_WARP_MASK, p_tilde, 0);

#pragma unroll
        for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
            const int dimi = lanei + ri * WARP_SIZE;
            if (dimi < head_dim) {
                z[ri] = alpha * z[ri] + p_tilde * v[ki * head_dim + dimi];
            }
        }
    }

    const float l_final = __shfl_sync(FULL_WARP_MASK, l, 0);
#pragma unroll
    for (int ri = 0; ri < DIMS_PER_LANE; ++ri) {
        const int dimi = ri * WARP_SIZE + lanei;
        if (dimi < head_dim) {
            o[qi * head_dim + dimi] = z[ri] / l_final;
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
    TORCH_CHECK(q.size(1) <= MAX_HEAD_DIM, "head dimension must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(q.numel() <= std::numeric_limits<int>::max(),
                "Q/K/V are too large for 32-bit kernel indexing");
}

template <int DIMS_PER_LANE>
void launch_forward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                    const int seq_len, const int head_dim, const float scale, const bool is_causal,
                    const dim3 grid, const dim3 block, const cudaStream_t stream) {
    fa2_forward_fp32_kernel<DIMS_PER_LANE>
        <<<grid, block, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                                     o.data_ptr<float>(), seq_len, head_dim, scale, is_causal);
}

constexpr std::array FORWARD_LAUNCHERS = {
    &launch_forward<1>,
    &launch_forward<2>,
    &launch_forward<3>,
    &launch_forward<4>,
};
static_assert(FORWARD_LAUNCHERS.size() == (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE);

} // namespace

at::Tensor fa2_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal) {
    check_inputs(q, k, v);
    const c10::cuda::CUDAGuard device_guard(q.device());

    const int seq_len = static_cast<int>(q.size(0));
    const int head_dim = static_cast<int>(q.size(1));
    auto o = at::empty_like(q);

    const dim3 grid(1 + (seq_len - 1) / WARPS_PER_BLOCK);
    const dim3 block(THREADS_PER_BLOCK);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int dims_per_lane = 1 + (head_dim - 1) / WARP_SIZE;
    FORWARD_LAUNCHERS[dims_per_lane - 1](
        q, k, v, o, seq_len, head_dim, scale, is_causal, grid, block, stream); // fmt: skip
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
