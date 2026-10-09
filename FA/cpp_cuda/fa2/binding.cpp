#include <torch/extension.h>

torch::Tensor fa2_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &fa2_forward_cuda,
      "FP32 online-softmax attention forward baseline (CUDA)");
}
