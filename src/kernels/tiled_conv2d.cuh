#include <cuda_runtime.h>

#define MASK_WIDTH 3
#define CONV2D_TILE_WIDTH 2



__global__ void tiled_conv2d(const float *A, float *output, const int C, const int H, const int W);
void wrap_test_tiled_conv2d();
