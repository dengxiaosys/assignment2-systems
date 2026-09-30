#include "fa/attention.h"
#include "fa/reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string(expression) + " failed at " + file + ':' + std::to_string(line) + ": " + cudaGetErrorString(error));
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

template<typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), count_(other.count_) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = other.count_;
        }
        return *this;
    }

    [[nodiscard]] T* data() {
        return data_;
    }

    [[nodiscard]] const T* data() const {
        return data_;
    }

    [[nodiscard]] std::size_t size() const {
        return count_;
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct Options {
    int query_count = 256;
    int key_count = 256;
    int head_dim = 64;
    int warmup_iterations = 5;
    int benchmark_iterations = 20;
    int seed = 0;
    bool causal = false;
    bool verify = true;
    std::string implementation = "both";
};

struct ErrorStats {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
    bool close = true;
};

using WorkspaceBytesFunction = std::size_t (*)(const fa::AttentionShape&);
using ForwardFunction = void (*)(const fa::ForwardParams&);
using BackwardFunction = void (*)(const fa::BackwardParams&);

struct Implementation {
    const char* name;
    WorkspaceBytesFunction forward_workspace_bytes;
    WorkspaceBytesFunction backward_workspace_bytes;
    ForwardFunction forward;
    BackwardFunction backward;
};

int parse_integer(const char* text, const char* option_name) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option_name) + " requires an integer");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next_value = [&](const char* option_name) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(option_name) + " requires a value");
            }
            return argv[++index];
        };

        if (argument == "--nq") {
            options.query_count = parse_integer(next_value("--nq"), "--nq");
        } else if (argument == "--nk") {
            options.key_count = parse_integer(next_value("--nk"), "--nk");
        } else if (argument == "--head-dim") {
            options.head_dim = parse_integer(next_value("--head-dim"), "--head-dim");
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_integer(next_value("--warmup"), "--warmup");
        } else if (argument == "--iterations") {
            options.benchmark_iterations = parse_integer(next_value("--iterations"), "--iterations");
        } else if (argument == "--seed") {
            options.seed = parse_integer(next_value("--seed"), "--seed");
        } else if (argument == "--impl") {
            options.implementation = next_value("--impl");
        } else if (argument == "--causal") {
            options.causal = true;
        } else if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument == "--help") {
            std::cout << "usage=fa_benchmark [--nq N] [--nk N] [--head-dim D] [--causal] "
                         "[--impl naive|tiled|both] [--warmup N] [--iterations N] [--seed N] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.warmup_iterations < 0 || options.benchmark_iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
    if (options.implementation != "naive" && options.implementation != "tiled" && options.implementation != "both") {
        throw std::invalid_argument("--impl must be naive, tiled, or both");
    }
    return options;
}

std::vector<float> random_vector(std::size_t count, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

template<typename T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (destination.size() != source.size()) {
        throw std::invalid_argument("host and device buffer sizes differ");
    }
    CUDA_CHECK(cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
std::vector<T> copy_to_host(const DeviceBuffer<T>& source) {
    std::vector<T> destination(source.size());
    CUDA_CHECK(cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return destination;
}

ErrorStats compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("cannot compare vectors with different sizes");
    }
    constexpr float absolute_tolerance = 5.0e-3F;
    constexpr float relative_tolerance = 5.0e-3F;
    ErrorStats stats;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            stats.max_absolute = std::numeric_limits<float>::infinity();
            stats.max_relative = std::numeric_limits<float>::infinity();
            stats.close = false;
            continue;
        }
        const float absolute_error = std::abs(actual[index] - expected[index]);
        const float relative_error = absolute_error / std::max(std::abs(expected[index]), 1.0e-6F);
        stats.max_absolute = std::max(stats.max_absolute, absolute_error);
        stats.max_relative = std::max(stats.max_relative, relative_error);
        stats.close = stats.close && absolute_error <= absolute_tolerance + relative_tolerance * std::abs(expected[index]);
    }
    return stats;
}

float benchmark_milliseconds(const std::function<void()>& launch, int warmup_iterations, int benchmark_iterations) {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_milliseconds / benchmark_iterations;
}

bool run_implementation(
    const Implementation& implementation,
    const Options& options,
    const fa::AttentionShape& shape,
    const std::vector<float>& host_query,
    const std::vector<float>& host_key,
    const std::vector<float>& host_value,
    const std::vector<float>& host_grad_output,
    const fa::HostForwardResult& reference_forward,
    const fa::HostBackwardResult& reference_backward) {
    DeviceBuffer<float> query(shape.query_elements());
    DeviceBuffer<float> key(shape.key_elements());
    DeviceBuffer<float> value(shape.key_elements());
    DeviceBuffer<float> output(shape.query_elements());
    DeviceBuffer<float> logsumexp(static_cast<std::size_t>(shape.query_count));
    DeviceBuffer<float> grad_output(shape.query_elements());
    DeviceBuffer<float> grad_query(shape.query_elements());
    DeviceBuffer<float> grad_key(shape.key_elements());
    DeviceBuffer<float> grad_value(shape.key_elements());

    const std::size_t forward_workspace_bytes = implementation.forward_workspace_bytes(shape);
    const std::size_t backward_workspace_bytes = implementation.backward_workspace_bytes(shape);
    DeviceBuffer<unsigned char> forward_workspace(forward_workspace_bytes);
    DeviceBuffer<unsigned char> backward_workspace(backward_workspace_bytes);

    copy_to_device(query, host_query);
    copy_to_device(key, host_key);
    copy_to_device(value, host_value);
    copy_to_device(grad_output, host_grad_output);

    const fa::ForwardParams forward_params{
        shape,
        query.data(),
        key.data(),
        value.data(),
        output.data(),
        logsumexp.data(),
        forward_workspace.data(),
        forward_workspace_bytes,
        nullptr,
    };
    const fa::BackwardParams backward_params{
        shape,
        query.data(),
        key.data(),
        value.data(),
        output.data(),
        logsumexp.data(),
        grad_output.data(),
        grad_query.data(),
        grad_key.data(),
        grad_value.data(),
        backward_workspace.data(),
        backward_workspace_bytes,
        nullptr,
    };

    implementation.forward(forward_params);
    implementation.backward(backward_params);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool verification_passed = true;
    if (options.verify) {
        const ErrorStats output_error = compare(copy_to_host(output), reference_forward.output);
        const ErrorStats logsumexp_error = compare(copy_to_host(logsumexp), reference_forward.logsumexp);
        const ErrorStats grad_query_error = compare(copy_to_host(grad_query), reference_backward.grad_query);
        const ErrorStats grad_key_error = compare(copy_to_host(grad_key), reference_backward.grad_key);
        const ErrorStats grad_value_error = compare(copy_to_host(grad_value), reference_backward.grad_value);
        verification_passed =
            output_error.close && logsumexp_error.close && grad_query_error.close && grad_key_error.close && grad_value_error.close;

        std::cout << "implementation=" << implementation.name << '\n';
        std::cout << "output_max_abs_error=" << output_error.max_absolute << '\n';
        std::cout << "logsumexp_max_abs_error=" << logsumexp_error.max_absolute << '\n';
        std::cout << "grad_query_max_abs_error=" << grad_query_error.max_absolute << '\n';
        std::cout << "grad_key_max_abs_error=" << grad_key_error.max_absolute << '\n';
        std::cout << "grad_value_max_abs_error=" << grad_value_error.max_absolute << '\n';
        std::cout << "verification_status=" << (verification_passed ? "passed" : "failed") << '\n';
    }

    const float forward_ms = benchmark_milliseconds(
        [&] {
            implementation.forward(forward_params);
        },
        options.warmup_iterations,
        options.benchmark_iterations);
    const float backward_ms = benchmark_milliseconds(
        [&] {
            implementation.backward(backward_params);
        },
        options.warmup_iterations,
        options.benchmark_iterations);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "implementation=" << implementation.name << '\n';
    std::cout << "forward_mean_ms=" << forward_ms << '\n';
    std::cout << "backward_mean_ms=" << backward_ms << '\n';
    std::cout << "forward_workspace_bytes=" << forward_workspace_bytes << '\n';
    std::cout << "backward_workspace_bytes=" << backward_workspace_bytes << '\n';
    return verification_passed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const fa::AttentionShape shape{options.query_count, options.key_count, options.head_dim, options.causal};
        shape.validate();

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::cout << "device_name=" << properties.name << '\n';
        std::cout << "compute_capability=" << properties.major << '.' << properties.minor << '\n';
        std::cout << "query_count=" << shape.query_count << '\n';
        std::cout << "key_count=" << shape.key_count << '\n';
        std::cout << "head_dim=" << shape.head_dim << '\n';
        std::cout << "causal=" << shape.causal << '\n';

        std::mt19937 generator(static_cast<std::mt19937::result_type>(options.seed));
        const auto query = random_vector(shape.query_elements(), generator);
        const auto key = random_vector(shape.key_elements(), generator);
        const auto value = random_vector(shape.key_elements(), generator);
        const auto grad_output = random_vector(shape.query_elements(), generator);
        fa::HostForwardResult forward_reference;
        fa::HostBackwardResult backward_reference;
        if (options.verify) {
            forward_reference = fa::reference_forward(shape, query, key, value);
            backward_reference =
                fa::reference_backward(shape, query, key, value, forward_reference.output, forward_reference.logsumexp, grad_output);
        }

        const Implementation naive{
            "naive",
            fa::naive_forward_workspace_bytes,
            fa::naive_backward_workspace_bytes,
            fa::naive_forward,
            fa::naive_backward,
        };
        const Implementation tiled{
            "tiled",
            fa::tiled_forward_workspace_bytes,
            fa::tiled_backward_workspace_bytes,
            fa::tiled_forward,
            fa::tiled_backward,
        };

        bool passed = true;
        if (options.implementation == "naive" || options.implementation == "both") {
            passed = run_implementation(
                         naive, options, shape, query, key, value, grad_output, forward_reference, backward_reference) &&
                     passed;
        }
        if (options.implementation == "tiled" || options.implementation == "both") {
            passed = run_implementation(
                         tiled, options, shape, query, key, value, grad_output, forward_reference, backward_reference) &&
                     passed;
        }
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_status=failed\n";
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
