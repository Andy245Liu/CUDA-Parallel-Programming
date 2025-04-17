#include <cuda_runtime.h>

#define GEMM_TILE_WIDTH 2

__global__ void tiled_GEMM(const float *A, const float *B, float *C, const int M, const int N, const int K);
void wrap_test_tiled_GEMM();
