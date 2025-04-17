#include "parallel_scan.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 4 // 4 is good for testing using small testcases here. We would want larger numbers for deployment (e.g., 1024)

#define MAX_NUM_BLOCKS 2048 //should be large enough to fit total number of blocks/ assuming maximum 2048*2048 entries, max 1024 threads per block, this gives us: 2048*2048/(2*1024) = 2048 blocks

__constant__ float OFFSETS[MAX_NUM_BLOCKS] ; //defining constant memory location to hold offsets calculated in the second kernel and used in the third kernel



__global__ void Brent_Kung_scan_kernel(const float *A, float *B, const int N){
    __shared__ float temp[2*BLOCK_SIZE];
    int stride = 1;
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    int idx = blockIdx.x * 2 * blockDim.x + index;
    int idx_small = blockIdx.x * 2 * blockDim.x + index - stride;
    temp[index] = (idx < N ? A[idx] : 0);
    temp[index - stride] = (idx_small < N ? A[idx_small] : 0);

    // printf("thread %d %f %f\n", threadIdx.x, temp[index], temp[index-stride]);
    while(stride < 2 * BLOCK_SIZE){
        __syncthreads();
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < N && index - stride >= 0){
            temp[index] += temp[index - stride];
        }
        stride <<= 1; // times 2
    }

    stride = BLOCK_SIZE >> 1; //divide by 2

    while(stride >= 1){
        __syncthreads();
        index = (threadIdx.x+1)*stride*2 - 1;
        if ((index+stride) < 2*BLOCK_SIZE){
            temp[index+stride] += temp[index];
        } 
        stride >>= 1; //divide by 2
    }
    __syncthreads();
    stride = 1;
    index = (threadIdx.x + 1) * stride * 2 - 1;
    if(idx < N){
        B[idx] = temp[index];
    }
    if(idx_small < N){
        B[idx_small] = temp[index-stride];
    }

    return;
}



__global__ void Brent_Kung_scan_blocksum_kernel(const float *A, float *B, const int original_array_size, const int num_blocks){
    __shared__ float temp[MAX_NUM_BLOCKS];
    int stride = 1;
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index - stride < num_blocks){
        int idx_small = min((index-stride)* 2 * BLOCK_SIZE + 2 * BLOCK_SIZE - 1, original_array_size-1);
        // printf("idx_small is: %d original last entry is: %d\n", idx_small, original_array_size-1);
        temp[index - stride] = A[idx_small];
    }

    if(index < num_blocks){
        int idx = min((index) * 2 * BLOCK_SIZE + 2 * BLOCK_SIZE - 1, original_array_size-1);
        // printf("IDX is: %d original last entry is: %d\n", idx, original_array_size-1);
        temp[index] = A[idx];
    }
    

    while(stride < num_blocks){//in this kernel, num_blocks is the equivalent of 2*block_size in the Brent_Kung_scan_kernel
        __syncthreads();
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < num_blocks && index - stride >= 0){
            temp[index] += temp[index - stride];
        }
        stride <<= 1; // times 2
    }

    stride = num_blocks >> 1; //stride set to half of output array size
    while(stride >= 1){
        __syncthreads();
        index = (threadIdx.x+1)*stride*2 - 1;
        if ((index+stride) < num_blocks){
            temp[index+stride] += temp[index];
        } 
        stride >>= 1; //divide by 2
    }
    __syncthreads();
    stride = 1;
    index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index - stride < num_blocks){
        B[index - stride] = temp[index - stride];
    }

    if(index < num_blocks){
        B[index] = temp[index];
    }
    return;
}




__global__ void Brent_Kung_scan_final_kernel(float *A, const int original_array_size){
    float offset = OFFSETS[blockIdx.x];
    int index = (threadIdx.x + 1) * 2 - 1;
    int idx = (blockIdx.x+1) * 2 * blockDim.x + index;
    int idx_small = (blockIdx.x+1) * 2 * blockDim.x + index - 1; 

    if(idx < original_array_size){
        A[idx] += offset;
    }
    if(idx_small < original_array_size){
        A[idx_small] += offset;
    }
   return; 
}

void wrap_test_parallel_scan(){
    int block_size = BLOCK_SIZE;
    int N = 19;

    float A[N] =  {1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
      
        10.0f, 11.0f, 12.0f, 
        13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f,
        19.5f};

    float B[N];

    float *d_a, *d_output;

    //copu data to device
    cudaMalloc((void**)&d_a, N*sizeof(float));
    cudaMalloc((void**)&d_output, N*sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, A,  N*sizeof(float), cudaMemcpyHostToDevice);
     


    //configuring kernel
    int num_blocks = ceil(1.0 * N / (2.0 * BLOCK_SIZE));
    dim3 dimGrid(num_blocks,1,1);
    dim3 dimBlock(block_size, 1,1);

    //******************* Launching first Brent-Kung scan CUDA kernel
    Brent_Kung_scan_kernel<<<dimGrid, dimBlock>>>(d_a, d_output, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(B, d_output,  N*sizeof(float), cudaMemcpyDeviceToHost);
 
    
    std::cout<<"Result of parallel scan first step:\n";    
    for(int i = 0; i <N; ++i){
       std::cout<<B[i]<<" ";
    }
    std::cout<<'\n';

    //******************* Launching second kernel for block sums
    float original_array_size = N;
    float block_sum_output[num_blocks];
    float *d_intermediate_output;
    cudaMalloc((void**)&d_intermediate_output, num_blocks*sizeof(float));
    dim3 dimGrid_intermediate(1,1,1);
    dim3 dimBlock_intermediate(ceil(1.0 *num_blocks / 2.0), 1,1);

    Brent_Kung_scan_blocksum_kernel<<<dimGrid_intermediate, dimBlock_intermediate>>>(d_output, d_intermediate_output, original_array_size, num_blocks);

    cudaDeviceSynchronize();
    cudaMemcpy(block_sum_output, d_intermediate_output,  num_blocks*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"Result of parallel scan second step:\n";    
    for(int i = 0; i <num_blocks; ++i){
       std::cout<<block_sum_output[i]<<" ";
    }
    std::cout<<'\n';


    //******************* Launching third kernel to get aggregated scan result across all initial blocks

    cudaMemcpyToSymbol(OFFSETS, block_sum_output, num_blocks*sizeof(float)); //results of second kernel copied to global memory for use in third kernel

    dim3 dimGrid_final(num_blocks-1,1,1);
    dim3 dimBlock_final(block_size, 1,1);

    Brent_Kung_scan_final_kernel<<<dimGrid_final, dimBlock_final>>>(d_output, original_array_size);


    cudaMemcpy(B, d_output,  N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"Final Result of parallel scan:\n";    
    for(int i = 0; i <N; ++i){
       std::cout<<B[i]<<" ";
    }
    std::cout<<'\n';


    // Free device memory
    cudaFree(d_a);
    cudaFree(d_output);
    cudaFree(d_intermediate_output);
    cudaFree(OFFSETS);
    
    
    return;
}