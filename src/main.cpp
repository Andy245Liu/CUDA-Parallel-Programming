#include <iostream>
#include "kernels/vector_add.cuh"
#include "kernels/tiled_GEMM.cuh"
#include "kernels/utils.cuh"
#include "kernels/tiled_conv2d.cuh"
#include "kernels/list_reduction.cuh"
#include "kernels/parallel_scan.cuh"
#include "kernels/histogram.cuh"
#include "kernels/sparse_matrix_multiply.cuh"

int main(){
    get_device_properties();

    std::cout<<"Running CUDA kernel tests\n";
    
    wrap_test_vector_add();
    wrap_test_tiled_GEMM();
    wrap_test_tiled_conv2d();
    wrap_test_list_reduction();
    wrap_test_parallel_scan();
    wrap_test_histogram();
    wrap_test_SpMV();

    return 0;
}