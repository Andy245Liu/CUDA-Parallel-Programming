cmake_minimum_required(VERSION 3.18)

project(cuda_parallel_programming
    VERSION 1.0
    DESCRIPTION "A brief CMake and CUDA experiment"
    LANGUAGES CXX CUDA)

add_executable(cuda_parallel_programming
    src/main.cpp
    src/kernels/vector_add.cu
    src/kernels/tiled_GEMM.cu
    src/kernels/tiled_conv2d.cu
    src/kernels/utils.cu
    src/kernels/list_reduction.cu
    src/kernels/parallel_scan.cu
    src/kernels/histogram.cu
    src/kernels/sparse_matrix_multiply.cu
    src/kernels/streams.cu
    src/sparse_matrix.cpp)

    
target_compile_features(cuda_parallel_programming PRIVATE cxx_std_20)
target_compile_options(cuda_parallel_programming PRIVATE -Wall -Wextra -Wpedantic)
