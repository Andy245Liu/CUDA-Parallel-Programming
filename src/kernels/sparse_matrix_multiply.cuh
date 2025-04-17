#include <cuda_runtime.h>
#include "../sparse_matrix.h"


#define MATRIX_SIZE 4
#define BLOCK_SIZE 2


__global__ void SpMV_kernel( JDS jds, float *vec, float* output, int M);
void wrap_test_SpMV();
