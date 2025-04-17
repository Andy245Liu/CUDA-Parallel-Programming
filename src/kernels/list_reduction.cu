#include "list_reduction.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 8

__global__ void reduction_sum_kernel(const float *A, float *B, const int N){
    __shared__ float temp[2 * BLOCK_SIZE];
    int stride = blockDim.x;
    int idx = blockIdx.x * 2*blockDim.x + threadIdx.x; //AL: TIMES 2 factor very important!!
    int idx_big = blockIdx.x * 2*blockDim.x + threadIdx.x + stride;
    temp[threadIdx.x]  = (idx < N ? A[idx] : 0);
    temp[threadIdx.x + stride]  = (idx_big < N ? A[idx_big] : 0);
    for(stride = blockDim.x; stride >= 1; stride >>= 1){
        __syncthreads();
        if(threadIdx.x < stride){
            // printf("sum here thread %d: %f + %f = %f\n", threadIdx.x, temp[threadIdx.x],  temp[threadIdx.x + stride],  temp[threadIdx.x + stride] +  temp[threadIdx.x]);
            temp[threadIdx.x] += temp[threadIdx.x + stride];
            
        }
    }
    
    if(threadIdx.x == 0){
        B[blockIdx.x] = temp[0];
    }
    return;
}
void wrap_test_list_reduction(){
    //For simplicity, you can assume that the input list will contain at most 2048 x 65535 elements so that it can be handled by only one kernel launch. 
    int block_size = BLOCK_SIZE;
    int N = 19;

    float A[N] =  {1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
      
        10.0f, 11.0f, 12.0f, 
        13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f,
        19.5f};

    int ret_array_size = (int)ceil(1.0*N /(2.0*block_size));
    float B[ret_array_size];

    float *d_a, *d_output;

    //copu data to device
    cudaMalloc((void**)&d_a, N*sizeof(float));
    cudaMalloc((void**)&d_output, ret_array_size*sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, A,  N*sizeof(float), cudaMemcpyHostToDevice);
     

    //configuring kernel
    dim3 dimGrid(ret_array_size,1,1);
    dim3 dimBlock(block_size, 1,1);

    // Launch CUDA kernel 
    reduction_sum_kernel<<<dimGrid, dimBlock>>>(d_a, d_output, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_output,  ret_array_size*sizeof(float), cudaMemcpyDeviceToHost);
 
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_output);
    
    float ans = 0;
    for(int i = 0; i <ret_array_size; ++i){
        ans += B[i];
    }
    std::cout<<"Result of list reduction: "<<ans<<'\n';    
    return;
}