#include "fa/reference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

float max_abs_difference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("cannot compare vectors with different sizes");
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        maximum = std::max(maximum, std::abs(lhs[index] - rhs[index]));
    }
    return maximum;
}

float loss(
    const fa::AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& grad_output) {
    const auto forward = fa::reference_forward(shape, query, key, value);
    double result = 0.0;
    for (std::size_t index = 0; index < forward.output.size(); ++index) {
        result += static_cast<double>(forward.output[index]) * grad_output[index];
    }
    return static_cast<float>(result);
}

std::vector<float> finite_difference(
    const fa::AttentionShape& shape,
    std::vector<float>& query,
    std::vector<float>& key,
    std::vector<float>& value,
    const std::vector<float>& grad_output,
    std::vector<float>& target) {
    constexpr float epsilon = 1.0e-3F;
    std::vector<float> gradient(target.size(), 0.0F);
    for (std::size_t index = 0; index < target.size(); ++index) {
        const float original = target[index];
        target[index] = original + epsilon;
        const float positive = loss(shape, query, key, value, grad_output);
        target[index] = original - epsilon;
        const float negative = loss(shape, query, key, value, grad_output);
        target[index] = original;
        gradient[index] = (positive - negative) / (2.0F * epsilon);
    }
    return gradient;
}

void run_case(bool causal) {
    const fa::AttentionShape shape{4, 4, 3, causal};
    std::vector<float> query{
        0.2F, -0.1F, 0.3F,
        -0.4F, 0.5F, 0.1F,
        0.7F, -0.2F, -0.3F,
        0.1F, 0.6F, -0.5F,
    };
    std::vector<float> key{
        -0.3F, 0.4F, 0.2F,
        0.6F, -0.2F, 0.1F,
        0.5F, 0.3F, -0.4F,
        -0.1F, -0.5F, 0.7F,
    };
    std::vector<float> value{
        0.4F, 0.1F, -0.2F,
        -0.3F, 0.8F, 0.2F,
        0.5F, -0.6F, 0.9F,
        0.7F, 0.2F, -0.1F,
    };
    const std::vector<float> grad_output{
        0.1F, -0.3F, 0.4F,
        0.2F, 0.5F, -0.2F,
        -0.6F, 0.7F, 0.3F,
        0.8F, -0.4F, 0.2F,
    };

    const auto forward = fa::reference_forward(shape, query, key, value);
    const auto backward = fa::reference_backward(shape, query, key, value, forward.output, forward.logsumexp, grad_output);
    const auto numerical_query = finite_difference(shape, query, key, value, grad_output, query);
    const auto numerical_key = finite_difference(shape, query, key, value, grad_output, key);
    const auto numerical_value = finite_difference(shape, query, key, value, grad_output, value);

    const float query_error = max_abs_difference(backward.grad_query, numerical_query);
    const float key_error = max_abs_difference(backward.grad_key, numerical_key);
    const float value_error = max_abs_difference(backward.grad_value, numerical_value);
    std::cout << "reference_case_causal=" << causal << '\n';
    std::cout << "grad_query_max_abs_error=" << query_error << '\n';
    std::cout << "grad_key_max_abs_error=" << key_error << '\n';
    std::cout << "grad_value_max_abs_error=" << value_error << '\n';

    constexpr float tolerance = 2.0e-3F;
    if (query_error > tolerance || key_error > tolerance || value_error > tolerance) {
        throw std::runtime_error("reference backward failed finite-difference validation");
    }
}

}  // namespace

int main() {
    try {
        run_case(false);
        run_case(true);
        std::cout << "reference_test_status=passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "reference_test_status=failed\n";
        std::cerr << "reference_test_error=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
