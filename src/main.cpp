#include <iostream>
#include "kernels/vector_add.cuh"
#include "kernels/tiled_GEMM.cuh"
#include "kernels/utils.cuh"
#include "kernels/tiled_conv2d.cuh"
#include "kernels/list_reduction.cuh"
#include "kernels/parallel_scan.cuh"
#include "kernels/histogram.cuh"
#include "kernels/sparse_matrix_multiply.cuh"
#include "kernels/streams.cuh"

int main(){
    get_device_properties();

    std::cout<<"Running CUDA kernel tests\n";
    std::cout<<"\nRunning CUDA kernel for vector addition\n";
    wrap_test_vector_add();
    std::cout<<"\nRunning CUDA kernel for tiled GEMM\n";
    wrap_test_tiled_GEMM();
    std::cout<<"\nRunning CUDA kernel for tiled Conv2D\n";
    wrap_test_tiled_conv2d();
    std::cout<<"\nRunning CUDA kernel for list reduction\n";
    wrap_test_list_reduction();
    std::cout<<"\nRunning CUDA kernel for parallel scan\n";
    wrap_test_parallel_scan();
    std::cout<<"\nRunning CUDA kernel for histogramming\n";
    wrap_test_histogram();
    std::cout<<"\nRunning CUDA kernel for sparse matrix vector multiplication\n";
    wrap_test_SpMV();
    std::cout<<"\nRunning CUDA kernel for CUDA streams\n";
    wrap_test_CUDA_streams();

    return 0;
}