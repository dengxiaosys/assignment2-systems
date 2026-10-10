cmake_minimum_required(VERSION 3.21)

project(cpp_add_demo LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cpp_add MODULE src/add.cpp)
target_compile_features(_cpp_add PRIVATE cxx_std_17)
target_compile_options(_cpp_add PRIVATE -Wall -Wextra -Wpedantic)

install(
    TARGETS _cpp_add
    LIBRARY DESTINATION cpp_add_demo
)
