cmake_minimum_required(VERSION 3.21)

project(cuda_flash_attention LANGUAGES CXX)

option(FA_ENABLE_CUDA "Build CUDA kernels and the GPU benchmark" ON)
option(FA_USE_FAST_MATH "Use CUDA fast approximate math in CUDA kernels" OFF)
set(FA_CUDA_ARCHITECTURE "61" CACHE STRING "CUDA architecture used for compilation")

add_library(fa_reference STATIC src/reference.cpp)
target_include_directories(fa_reference PUBLIC include)
target_compile_features(fa_reference PUBLIC cxx_std_17)
target_compile_options(fa_reference PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fa_reference_test tests/reference_test.cpp)
target_link_libraries(fa_reference_test PRIVATE fa_reference)

enable_testing()
add_test(NAME fa_reference_test COMMAND fa_reference_test)

if(FA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "FA_ENABLE_CUDA=ON requires nvcc; configure with -DFA_ENABLE_CUDA=OFF for CPU reference tests only")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(
        fa_cuda
        STATIC
        src/naive.cu
        src/tiled_forward.cu
        src/tiled_backward.cu
    )
    target_include_directories(fa_cuda PUBLIC include PRIVATE src)
    target_compile_features(fa_cuda PUBLIC cxx_std_17)
    set_target_properties(
        fa_cuda
        PROPERTIES
        CUDA_ARCHITECTURES "${FA_CUDA_ARCHITECTURE}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_options(fa_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    if(FA_USE_FAST_MATH)
        target_compile_options(fa_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    target_link_libraries(fa_cuda PUBLIC CUDA::cudart)

    add_executable(fa_benchmark app/benchmark.cu)
    target_include_directories(fa_benchmark PRIVATE include)
    target_compile_features(fa_benchmark PRIVATE cxx_std_17)
    set_target_properties(
        fa_benchmark
        PROPERTIES
        CUDA_ARCHITECTURES "${FA_CUDA_ARCHITECTURE}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(fa_benchmark PRIVATE fa_cuda fa_reference CUDA::cudart)
endif()
