#include "histogram.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>




__global__ void cast_img_to_unsigned_char(const float *A,  uchar *B, const int C, const int H, const int W){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if(idx < W && idy < H && idz < C){
        B[H*W*idz + W * idy + idx] = (uchar) A[H*W*idz + W * idy + idx];
    }
    return;    
}


__global__ void rgb_to_grayscale(const uchar *A,  uchar *B, const int H, const int W){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < W && idy < H){
        uchar  r = A[H*W*0 + W * idy + idx];
        uchar  g = A[H*W*1 + W * idy + idx];
        uchar  b = A[H*W*2 + W * idy + idx];
        B[W * idy + idx] = (uchar) (0.21 * r + 0.72 * g + 0.07 * b);
    }
    return;    
}

__global__ void histogram_kernel(const uchar *grayscale_image, uint *histo, int H, int W){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //This will not work correctly if there are fewer than 256 threads inside the block
    __shared__ uint histo_private[PIXEL_VALUE_RANGE];

    int thread_id = blockDim.x * threadIdx.y +threadIdx.x;
    if(thread_id < PIXEL_VALUE_RANGE){
        histo_private[thread_id] = 0;
    }

    __syncthreads();


    if(idx < W && idy < H){
        atomicAdd(&(histo_private[grayscale_image[W * idy + idx]]),1);

    }
    __syncthreads();

    if(thread_id < PIXEL_VALUE_RANGE){
        atomicAdd(&(histo[thread_id]),histo_private[thread_id]);
    }

}



__global__ void histogram_CDF_kernel(const uint *histo, float* cdf, int H, int W){
    __shared__ float temp[PIXEL_VALUE_RANGE];
    int stride = 1;
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    int idx = blockIdx.x * 2 * blockDim.x + index;
    int idx_small = blockIdx.x * 2 * blockDim.x + index - stride;

    temp[index] = (idx < PIXEL_VALUE_RANGE ? (1.0* histo[idx])/(H*W) : 0);
    temp[index - stride] = (idx_small < PIXEL_VALUE_RANGE ? (1.0*histo[idx_small])/(H*W) : 0);

    __syncthreads();

    while(stride <  PIXEL_VALUE_RANGE){
        __syncthreads();
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < PIXEL_VALUE_RANGE && index - stride >= 0){
            temp[index] += temp[index - stride] ;
        }
        stride <<= 1; // times 2
    }

    stride = PIXEL_VALUE_RANGE >> 2; //divide by 4

    while(stride >= 1){
        __syncthreads();
        index = (threadIdx.x+1)*stride*2 - 1;
        if ((index+stride) < PIXEL_VALUE_RANGE){
            temp[index+stride] += temp[index];
        } 
        stride >>= 1; //divide by 2
    }
    __syncthreads();

    stride = 1;
    index = (threadIdx.x + 1) * stride * 2 - 1;
    if(idx < PIXEL_VALUE_RANGE){
        cdf[idx] = temp[index];
    }
    if(idx_small < PIXEL_VALUE_RANGE){
        cdf[idx_small] = temp[index-stride];
    }

    return;
}

__global__ void correct_pixel_kernel(const uchar *unsigned_char_rgd_img, const float* cdf, float *output_img, const int C, const int H, const int W){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if(idx < W && idy < H && idz < C){
        uchar pixel_value = unsigned_char_rgd_img[H*W*idz + W * idy + idx] ;
        float corrected_pixel =  min(max(255*(cdf[pixel_value]-cdf[0])/ (1.0 - cdf[0]) , 0.0), 255.0);
        output_img[H*W*idz + W * idy + idx] = (float) (corrected_pixel/ 255.0);   
        // printf("thead idx: %d %d %d pixel_val: %u, corrected = %f\n", idx, idy, idz, pixel_value, output_img[H*W*idz + W * idy + idx]);
    }
    return;    
}




void wrap_test_histogram(){

    const int C = 3;
    const int H = 3;
    const int W = 3;
  
    float A[C*H*W] =  {1.0f, 2.0f, 3.0f, 
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,
  
    1.0f, 2.0f, 3.0f, 
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,
  
    1.0f, 2.0f, 3.0f, 
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f};

    uchar B[C*H*W];
    //**************** Float to unsigned char step
    // Device memory
    float *d_a;
    uchar *d_a_uchar_cast;
    cudaMalloc((void**)&d_a, C*H*W*sizeof(float));
    cudaMalloc((void**)&d_a_uchar_cast, C*H*W*sizeof(uchar));

    // Copy input data to device
    cudaMemcpy(d_a, A,  C*H*W*sizeof(float), cudaMemcpyHostToDevice);

    //configuring kernel
    dim3 dimGrid(ceil((1.0*W)/BLOCK_SIZE),ceil((1.0*H)/BLOCK_SIZE), ceil((1.0*C)/BLOCK_SIZE));//AL: Be careful of x, y z dimenions! Ordering matters
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    // Launch CUDA kernel 
    cast_img_to_unsigned_char<<<dimGrid, dimBlock>>>(d_a, d_a_uchar_cast, C, H, W);
  
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_a_uchar_cast,  C*H*W*sizeof(uchar), cudaMemcpyDeviceToHost);

    std::cout << "Result of casting:\n[\n" << std::endl;
  
    for(int i = 0; i < C; ++i){
        std::cout<<"[\n";
        for(int j = 0; j < H; ++j){
          std::cout<<"[ ";
          for(int k = 0; k < W; ++k){
            std::cout<<static_cast<unsigned>(B[i * H * W + j * W +k])<<" ";
          }
          std::cout<<"]\n";
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";
 

   //**************** RGB to grayscale step
   uchar grayscale[H*W];
   uchar *d_grayscale;

   cudaMalloc((void**)&d_grayscale, H*W*sizeof(uchar));

   dim3 dimGrid_grayscale(ceil((1.0*W)/BLOCK_SIZE),ceil((1.0*H)/BLOCK_SIZE), 1);//AL: Be caareful of x, y z dimenions! Ordering matters
   dim3 dimBlock_grayscale(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Launch CUDA kernel 
    rgb_to_grayscale<<<dimGrid_grayscale, dimBlock_grayscale>>>(d_a_uchar_cast, d_grayscale, H, W);
    cudaDeviceSynchronize();
    cudaMemcpy(grayscale, d_grayscale,  H*W*sizeof(uchar), cudaMemcpyDeviceToHost);

    std::cout << "Result of grayscale conversion:\n[\n" << std::endl;
  
 
    for(int j = 0; j < H; ++j){
        std::cout<<"[ ";
        for(int k = 0; k < W; ++k){
        uchar u = grayscale[j * W +k];
        printf("%u ", u);
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";

    //**************** Grayscale to Histogram step
    uint histo[PIXEL_VALUE_RANGE];
    uint *d_histo;
    cudaMalloc((void**)&d_histo, PIXEL_VALUE_RANGE*sizeof(uint));
    int thread_min_dim_for_hist = 16;

    dim3 dimGrid_histo(ceil((1.0*W)/thread_min_dim_for_hist),ceil((1.0*H)/thread_min_dim_for_hist), 1);//AL: Be caareful of x, y z dimenions! Ordering matters
    dim3 dimBlock_histo(thread_min_dim_for_hist, thread_min_dim_for_hist, 1);

    histogram_kernel<<<dimGrid_histo, dimBlock_histo>>>(d_grayscale, d_histo, H, W);

    cudaDeviceSynchronize();
    cudaMemcpy(histo, d_histo,  PIXEL_VALUE_RANGE*sizeof(uint), cudaMemcpyDeviceToHost);

    std::cout << "First 15 results of histogram: " << std::endl;
    for(int i = 0; i < 15; ++i){
        std::cout<<histo[i]<<" ";
    }
    std::cout<<'\n';


    //**************** Histogram CDF step
    // histogram_CDF_kernel(const uint *histo, float* cdf, int H, int W)

    float cdf[PIXEL_VALUE_RANGE];
    float *d_cdf;
    cudaMalloc((void**)&d_cdf, PIXEL_VALUE_RANGE*sizeof(float));

    dim3 dimGrid_cdf(1, 1, 1);//AL: Be caareful of x, y z dimenions! Ordering matters
    dim3 dimBlock_cdf(PIXEL_VALUE_RANGE/2, 1, 1); //only need half the number of threads as dim of histogram

    histogram_CDF_kernel<<<dimGrid_cdf, dimBlock_cdf>>>(d_histo, d_cdf, H, W);

    cudaDeviceSynchronize();
    cudaMemcpy(cdf, d_cdf,  PIXEL_VALUE_RANGE*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 15 results of histogram CDF: " << std::endl;
    for(int i = 0; i < 15; ++i){
        std::cout<<cdf[i]<<" ";
    }
    std::cout<<'\n';




    //**************** Apply Histogram Equalization and Cast RGB Image back to float

    float output[C*H*W];
    float *d_output;
    cudaMalloc((void**)&d_output, C*H*W*sizeof(float));


    //configuring kernel
    dim3 dimGrid_final(ceil((1.0*W)/BLOCK_SIZE),ceil((1.0*H)/BLOCK_SIZE), ceil((1.0*C)/BLOCK_SIZE));//AL: Be careful of x, y z dimenions! Ordering matters
    dim3 dimBlock_final(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    // Launch CUDA kernel 
    correct_pixel_kernel<<<dimGrid_final, dimBlock_final>>>(d_a_uchar_cast, d_cdf, d_output, C, H, W);
  
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output,  C*H*W*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result of casting:\n[\n" << std::endl;
  
    for(int i = 0; i < C; ++i){
        std::cout<<"[\n";
        for(int j = 0; j < H; ++j){
          std::cout<<"[ ";
          for(int k = 0; k < W; ++k){
            std::cout<<(output[i * H * W + j * W +k])<<" ";
          }
          std::cout<<"]\n";
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";


   //Freeing CUDA Device Memory
   cudaFree(d_a);
   cudaFree(d_a_uchar_cast);
   cudaFree(d_grayscale);
   cudaFree(d_histo);
   cudaFree(d_cdf);
   cudaFree(d_output);

   return;


}

