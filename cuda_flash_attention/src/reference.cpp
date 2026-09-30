#include "fa/reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fa {
namespace {

void require_size(const std::vector<float>& values, std::size_t expected, const char* name) {
    if (values.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has an unexpected number of elements");
    }
}

std::size_t offset(int row, int column, int row_width) {
    return static_cast<std::size_t>(row) * row_width + column;
}

double score_at(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    int query_index,
    int key_index) {
    double score = 0.0;
    for (int feature = 0; feature < shape.head_dim; ++feature) {
        score += static_cast<double>(query[offset(query_index, feature, shape.head_dim)]) *
                 static_cast<double>(key[offset(key_index, feature, shape.head_dim)]);
    }
    return score / std::sqrt(static_cast<double>(shape.head_dim));
}

}  // namespace

HostForwardResult reference_forward(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value) {
    shape.validate();
    require_size(query, shape.query_elements(), "query");
    require_size(key, shape.key_elements(), "key");
    require_size(value, shape.key_elements(), "value");

    HostForwardResult result{
        std::vector<float>(shape.query_elements(), 0.0F),
        std::vector<float>(static_cast<std::size_t>(shape.query_count), 0.0F),
    };
    std::vector<double> scores(static_cast<std::size_t>(shape.key_count));

    for (int query_index = 0; query_index < shape.query_count; ++query_index) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (int key_index = 0; key_index < shape.key_count; ++key_index) {
            const bool visible = !shape.causal || key_index <= query_index;
            const double score = visible ? score_at(shape, query, key, query_index, key_index)
                                         : -std::numeric_limits<double>::infinity();
            scores[static_cast<std::size_t>(key_index)] = score;
            row_max = std::max(row_max, score);
        }

        double denominator = 0.0;
        for (int key_index = 0; key_index < shape.key_count; ++key_index) {
            denominator += std::exp(scores[static_cast<std::size_t>(key_index)] - row_max);
        }
        const double logsumexp = row_max + std::log(denominator);
        result.logsumexp[static_cast<std::size_t>(query_index)] = static_cast<float>(logsumexp);

        for (int feature = 0; feature < shape.head_dim; ++feature) {
            double output = 0.0;
            for (int key_index = 0; key_index < shape.key_count; ++key_index) {
                const double probability = std::exp(scores[static_cast<std::size_t>(key_index)] - logsumexp);
                output += probability * value[offset(key_index, feature, shape.head_dim)];
            }
            result.output[offset(query_index, feature, shape.head_dim)] = static_cast<float>(output);
        }
    }
    return result;
}

HostBackwardResult reference_backward(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& output,
    const std::vector<float>& logsumexp,
    const std::vector<float>& grad_output) {
    shape.validate();
    require_size(query, shape.query_elements(), "query");
    require_size(key, shape.key_elements(), "key");
    require_size(value, shape.key_elements(), "value");
    require_size(output, shape.query_elements(), "output");
    require_size(logsumexp, static_cast<std::size_t>(shape.query_count), "logsumexp");
    require_size(grad_output, shape.query_elements(), "grad_output");

    HostBackwardResult result{
        std::vector<float>(shape.query_elements(), 0.0F),
        std::vector<float>(shape.key_elements(), 0.0F),
        std::vector<float>(shape.key_elements(), 0.0F),
    };
    std::vector<double> grad_query_accumulator(shape.query_elements(), 0.0);
    std::vector<double> grad_key_accumulator(shape.key_elements(), 0.0);
    std::vector<double> grad_value_accumulator(shape.key_elements(), 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(shape.head_dim));

    for (int query_index = 0; query_index < shape.query_count; ++query_index) {
        double correction = 0.0;
        for (int feature = 0; feature < shape.head_dim; ++feature) {
            correction += static_cast<double>(output[offset(query_index, feature, shape.head_dim)]) *
                          grad_output[offset(query_index, feature, shape.head_dim)];
        }

        for (int key_index = 0; key_index < shape.key_count; ++key_index) {
            if (shape.causal && key_index > query_index) {
                continue;
            }
            const double score = score_at(shape, query, key, query_index, key_index);
            const double probability = std::exp(score - logsumexp[static_cast<std::size_t>(query_index)]);

            double grad_probability = 0.0;
            for (int feature = 0; feature < shape.head_dim; ++feature) {
                grad_probability += static_cast<double>(grad_output[offset(query_index, feature, shape.head_dim)]) *
                                    value[offset(key_index, feature, shape.head_dim)];
            }
            const double grad_score = probability * (grad_probability - correction);

            for (int feature = 0; feature < shape.head_dim; ++feature) {
                grad_value_accumulator[offset(key_index, feature, shape.head_dim)] +=
                    probability * grad_output[offset(query_index, feature, shape.head_dim)];
                grad_query_accumulator[offset(query_index, feature, shape.head_dim)] +=
                    grad_score * key[offset(key_index, feature, shape.head_dim)] * scale;
                grad_key_accumulator[offset(key_index, feature, shape.head_dim)] +=
                    grad_score * query[offset(query_index, feature, shape.head_dim)] * scale;
            }
        }
    }

    for (std::size_t index = 0; index < result.grad_query.size(); ++index) {
        result.grad_query[index] = static_cast<float>(grad_query_accumulator[index]);
    }
    for (std::size_t index = 0; index < result.grad_key.size(); ++index) {
        result.grad_key[index] = static_cast<float>(grad_key_accumulator[index]);
        result.grad_value[index] = static_cast<float>(grad_value_accumulator[index]);
    }
    return result;
}

}  // namespace fa
