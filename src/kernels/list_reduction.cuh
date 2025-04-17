#include <cuda_runtime.h>

__global__ void reduction_sum_kernel(const float *A, float *B, const int N);
void wrap_test_list_reduction();