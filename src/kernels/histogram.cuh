#include <cuda_runtime.h>

#define BLOCK_SIZE 2
#define PIXEL_VALUE_RANGE 256
typedef unsigned char uchar;
typedef unsigned int uint;

__global__ void rgb_to_grayscale(const uchar *A,  uchar *B, const int H, const int W);
__global__ void rgb_to_grayscale(const uchar *A,  uchar *B, const int H, const int W);
__global__ void histogram_kernel(const uchar *grayscale_image, uint *histo, int H, int W);
__global__ void histogram_CDF_kernel(const uint *histo, float* cdf, int H, int W);
__global__ void correct_pixel_kernel(const uchar *unsigned_char_rgd_img, const float* cdf, float *output_img, const int C, const int H, const int W);

void wrap_test_histogram();