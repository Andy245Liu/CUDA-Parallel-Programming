#include <cuda_runtime.h>


__global__ void Brent_Kung_scan_kernel(const float *A, float *B, const int N);
__global__ void Brent_Kung_scan_blocksum_kernel(const float *A, float *B,  const int original_array_size, const int num_blocks);
__global__ void Brent_Kung_scan_final_kernel(float *A,  const int original_array_size);    
void wrap_test_parallel_scan();