#include <cuda_runtime.h>

__global__ void tiled_GEMM(const float *A, const float *B, float *C, const int M, const int N, const int K);
void wrap_test_tiled_GEMM();
