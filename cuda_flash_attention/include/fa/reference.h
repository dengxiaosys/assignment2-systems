#pragma once

#include <vector>

#include "fa/shape.h"

namespace fa {

struct HostForwardResult {
    std::vector<float> output;
    std::vector<float> logsumexp;
};

struct HostBackwardResult {
    std::vector<float> grad_query;
    std::vector<float> grad_key;
    std::vector<float> grad_value;
};

HostForwardResult reference_forward(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value);

HostBackwardResult reference_backward(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& output,
    const std::vector<float>& logsumexp,
    const std::vector<float>& grad_output);

}  // namespace fa
