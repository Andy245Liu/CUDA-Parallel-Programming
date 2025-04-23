
#include "tiled_GEMM.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define GEMM_TILE_WIDTH 2

__global__ void tiled_GEMM(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    //This performs GEMM on A * B = C, where A is size M x K and B is size K x N
    __shared__ float subTileA[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    __shared__ float subTileB[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * GEMM_TILE_WIDTH + ty;
    int Col = bx * GEMM_TILE_WIDTH + tx;
    float Cvalue = 0;


    for (int q = 0; q < (K-1)/GEMM_TILE_WIDTH + 1; ++q) {
        
            // Collaborative loading of M and N tiles into shared memory
            if(Row < M &&   q*GEMM_TILE_WIDTH+tx < K ){
                subTileA[ty][tx] = A[Row*K + q*GEMM_TILE_WIDTH+tx];
            }
            else{
                subTileA[ty][tx] = 0;
            }

            if(q*GEMM_TILE_WIDTH+ty < K && Col < N ){
                subTileB[ty][tx] = B[(q*GEMM_TILE_WIDTH+ty)*N+Col];
            }
            else{
                subTileB[ty][tx] = 0;
            }
           
            __syncthreads();
            for (int k = 0; k < GEMM_TILE_WIDTH; ++k){
                Cvalue += subTileA[ty][k] * subTileB[k][tx];
            }
            __syncthreads();
            
    }
    if(Row < M && Col < N){
        C[Row*N+Col] = Cvalue;
     }   
    return;
}


void wrap_test_tiled_GEMM(){
    int M = 3;
    int N = 7;
    int K = 5;
    
    float h_a[M][K] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 
                       {2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                       {0.0f, 2.0f, 3.0f, 4.0f, 5.0f}
                    };
    float h_b[K][N] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, 
                       {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f},
                       {0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
                       {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f},
                       {0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}
                      };
  
    float h_c[M][N];
  
    // Device memory
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, M*K*sizeof(float));
    cudaMalloc((void**)&d_b, K*N*sizeof(float));
    cudaMalloc((void**)&d_c, M*N*sizeof(float));
  
    // Copy input data to device
    cudaMemcpy(d_a, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice);
  
    //configuring kernel
    dim3 dimGrid(ceil((1.0*N)/GEMM_TILE_WIDTH),ceil((1.0*M)/GEMM_TILE_WIDTH), 1);//AL: Be caareful of x, y z dimenions! Ordering matters
    dim3 dimBlock(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
  
    // Launch CUDA kernel 
    tiled_GEMM<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N , K);
  
    // Copy result back to host
    cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  
    
  
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  
    // Print results
    std::cout << "Result of Tiled GEMM:\n[" << std::endl;
  
    for(int i = 0; i < M; ++i){
        std::cout<<"[ ";
        for(int j = 0; j < N; ++j){
            std::cout<<h_c[i][j]<<" ";
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";
  
    return;
  }
  