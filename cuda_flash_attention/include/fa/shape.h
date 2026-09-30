#pragma once

#include <cstddef>
#include <stdexcept>

namespace fa {

struct AttentionShape {
    int query_count;
    int key_count;
    int head_dim;
    bool causal;

    void validate() const {
        if (query_count <= 0 || key_count <= 0 || head_dim <= 0) {
            throw std::invalid_argument("query_count, key_count, and head_dim must be positive");
        }
    }

    [[nodiscard]] std::size_t query_elements() const {
        validate();
        return static_cast<std::size_t>(query_count) * head_dim;
    }

    [[nodiscard]] std::size_t key_elements() const {
        validate();
        return static_cast<std::size_t>(key_count) * head_dim;
    }

    [[nodiscard]] std::size_t score_elements() const {
        validate();
        return static_cast<std::size_t>(query_count) * key_count;
    }
};

}  // namespace fa
