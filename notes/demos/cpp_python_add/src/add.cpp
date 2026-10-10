#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::int64_t add(std::int64_t a, std::int64_t b) {
    constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
    if ((b > 0 && a > max_value - b) || (b < 0 && a < min_value - b)) {
        throw std::overflow_error("int64 addition overflow");
    }
    return a + b;
}

}  // namespace

PYBIND11_MODULE(_cpp_add, module) {
    module.doc() = "A C++ addition function exposed through pybind11";

    module.def(
        "add",
        &add,
        py::arg("a"),
        py::arg("b"),
        "Add two C++ int64 values with overflow checking.");
}
