
#include "tiled_conv2d.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

//@@ Define constant memory for device convolution kernel here
__constant__ float MASK_conv2d[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH] ;

__global__ void tiled_conv2d(const float *A, float *output, const int C, const int H, const int W){
    const int Kx = MASK_WIDTH;
    const int Ky = MASK_WIDTH;
    const int Kz = MASK_WIDTH;
    int MASK_RADIUS_x = Kx / 2;
    int MASK_RADIUS_y = Ky / 2;
    int MASK_RADIUS_z = Kz / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    __shared__ float tile[CONV2D_TILE_WIDTH] [CONV2D_TILE_WIDTH] [CONV2D_TILE_WIDTH];
    if(x >= 0 && x < W && y  >= 0 && y < H && z >= 0 && z < C){
        tile[threadIdx.z] [threadIdx.y] [threadIdx.x] = A[z*H*W + y * W + x];
    } else{
        tile[threadIdx.z] [threadIdx.y] [threadIdx.x] = 0;
    }
    __syncthreads();


    if(x >= 0 && x < W && y  >= 0 && y < H && z >= 0 && z < C){
        int curr_tile_x_start = blockIdx.x * blockDim.x;
        int curr_tile_y_start = blockIdx.y * blockDim.y;
        int curr_tile_z_start = blockIdx.z * blockDim.z;

        int next_tile_x_start = (blockIdx.x+1) * blockDim.x;
        int next_tile_y_start = (blockIdx.y+1) * blockDim.y;
        int next_tile_z_start = (blockIdx.z+1) * blockDim.z;
        float Pvalue = 0;
        for(int k = -MASK_RADIUS_z; k <= MASK_RADIUS_z; ++k){
            for(int j = -MASK_RADIUS_y; j <= MASK_RADIUS_y; ++j){
                for(int i = -MASK_RADIUS_x; i <= MASK_RADIUS_x; ++i){
                   
                    if(x + i >= 0 && x + i < W && y + j >=0 && y + j < H && z + k >=0 && z+k < C){
                        
                        if(x + i >= curr_tile_x_start && x + i < next_tile_x_start && y + j >=curr_tile_y_start && y + j < next_tile_y_start && z + k >= curr_tile_z_start&& z+k < next_tile_z_start){
                           
                            Pvalue += tile[(threadIdx.z + k) ][(threadIdx.y + j)] [(threadIdx.x + i)] * MASK_conv2d[(k+MASK_RADIUS_z) * Kx*Ky + (j+MASK_RADIUS_y) * Kx + i+MASK_RADIUS_x];
                       
                        } else{
                            Pvalue += A[(z+k) * H*W + (y+j) * W + x+i] * MASK_conv2d[(k+MASK_RADIUS_z) * Kx*Ky + (j+MASK_RADIUS_y) * Kx + i+MASK_RADIUS_x];

                        }
                    }
                }
            }

        }
        output[z*H*W + y * W + x] = Pvalue;
    }
    __syncthreads();


    return;
}


void wrap_test_tiled_conv2d(){
    const int C = 3;
    const int H = 3;
    const int W = 3;
  
    float A[C*H*W] =  {
        1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    
        1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    
        1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };    
  
    float mask[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH] = {
        1.0f, 2.0f, 1.0f, 
        0.0f, 0.0f, 0.0f,
       -1.0f, -2.0f, -1.0f,
  
       1.0f, 2.0f, 1.0f, 
       0.0f, 0.0f, 0.0f,
      -1.0f, -2.0f, -1.0f,
  
        1.0f, 2.0f, 1.0f, 
        0.0f, 0.0f, 0.0f,
       -1.0f, -2.0f, -1.0f
    };

  
    float output[C*H*W];
  
    // Device memory
    
    float *d_a, *d_output;
    cudaMalloc((void**)&d_a, C*H*W*sizeof(float));
    cudaMalloc((void**)&d_output, C*H*W*sizeof(float));
  
    // Copy input data to device
    cudaMemcpy(d_a, A,  C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK_conv2d, mask, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float)); //kernel copied to constant memory. it's fast and shared among all blocks
  
    //configuring kernel
    dim3 dimGrid(ceil((1.0*W)/CONV2D_TILE_WIDTH),ceil((1.0*H)/CONV2D_TILE_WIDTH), ceil((1.0*C)/CONV2D_TILE_WIDTH));//AL: Be caareful of x, y z dimenions! Ordering matters
    dim3 dimBlock(CONV2D_TILE_WIDTH, CONV2D_TILE_WIDTH, CONV2D_TILE_WIDTH);
  
     // Launch CUDA kernel 
     tiled_conv2d<<<dimGrid, dimBlock>>>(d_a, d_output, C, H, W);
  
     cudaDeviceSynchronize();
     cudaMemcpy(output, d_output,  C*H*W*sizeof(float), cudaMemcpyDeviceToHost);
  
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_output);
    cudaFree(MASK_conv2d);
  
    // Print results
    std::cout << "Result of Conv2D:\n[\n" << std::endl;
  
    for(int i = 0; i < C; ++i){
        std::cout<<"[\n";
        for(int j = 0; j < H; ++j){
          std::cout<<"[ ";
          for(int k = 0; k < W; ++k){
            std::cout<<output[i * H * W + j * W +k]<<" ";
          }
          std::cout<<"]\n";
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";
  
    
    return;
  }