#include <cuda_runtime.h>

__global__ void add_vectors(const float *a, const float *b, float *c, const int N);
void wrap_test_vector_add();
