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

constexpr int kThreadsPerBlock = 128;
constexpr int kMaxHeadDim = kThreadsPerBlock;

__global__ void fa2_forward_fp32_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                        const float *__restrict__ v, float *const __restrict__ o,
                                        const int seq_len, const int head_dim, const float scale,
                                        const bool is_causal) {
    __shared__ float reduction[kThreadsPerBlock];
    __shared__ float alpha;    // exp(m_old - m_new)
    __shared__ float p_tilde;  // exp(s - m_new), unnormalized weight
    __shared__ float l_shared; // Final l broadcast from thread 0.

    const int qi = blockIdx.x;
    const int dim = threadIdx.x;
    const bool is_valid_dim = dim < head_dim;
    const float q_dim = is_valid_dim ? q[qi * head_dim + dim] : 0.0f;
    float z = 0.0f; // This thread's component of the numerator vector z.

    float m = -CUDART_INF_F; // Running maximum; updated by thread 0.
    float l = 0.0f;          // Denominator ell in the textbook; thread 0.
    const int key_limit = is_causal ? qi + 1 : seq_len;

    for (int ki = 0; ki < key_limit; ++ki) {
        reduction[dim] = is_valid_dim ? q_dim * k[ki * head_dim + dim] : 0.0f;
        __syncthreads();

        for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
            if (dim < stride) {
                reduction[dim] += reduction[dim + stride];
            }
            __syncthreads();
        }

        // online softmax
        if (dim == 0) {
            const float s = reduction[0] * scale;
            const float m_new = fmaxf(m, s);
            alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
            p_tilde = expf(s - m_new);
            l = alpha * l + p_tilde;

            m = m_new;
        }
        __syncthreads();

        if (is_valid_dim) {
            z = alpha * z + p_tilde * v[ki * head_dim + dim];
        }
        __syncthreads();
    }

    if (dim == 0) {
        l_shared = l;
    }
    __syncthreads();
    if (is_valid_dim) {
        o[qi * head_dim + dim] = z / l_shared;
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

} // namespace

at::Tensor fa2_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal) {
    check_inputs(q, k, v);
    const c10::cuda::CUDAGuard device_guard(q.device());

    const int seq_len = static_cast<int>(q.size(0));
    const int head_dim = static_cast<int>(q.size(1));
    auto o = at::empty_like(q);

    const dim3 grid(seq_len);
    const dim3 block(kThreadsPerBlock);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fa2_forward_fp32_kernel<<<grid, block, 0, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(),
                                                        v.data_ptr<float>(), o.data_ptr<float>(), seq_len,
                                                        head_dim, scale, is_causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
