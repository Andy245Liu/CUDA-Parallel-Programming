#include "streams.cuh"
#include "tiled_GEMM.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <tuple>


#define GEMM_TILE_WIDTH 2

void print_matrix(float *C, size_t M, size_t N){
    std::cout << "Result of Tiled GEMM:\n[" << std::endl;
  
    for(size_t i = 0; i < M; ++i){
        std::cout<<"[ ";
        for(size_t j = 0; j < N; ++j){
            std::cout<<C[i*N + j]<<" ";
        }
        std::cout<<"]\n";
    }
    std::cout<<"]\n";
    return;
}
void wrap_test_CUDA_streams(){
    int stream_step = 3;
    std::vector<std::tuple<size_t,size_t,size_t>> dimvec;
    std::vector<std::tuple<float*,float*,float*>> array_ptr_vec_device;
    std::vector<std::tuple<float*,float*,float*>> array_ptr_vec_host;

    //initializing 0th GEMM operation
    size_t M0 = 3;
    size_t N0 = 7;
    size_t K0 = 5;

    float h_a0[M0*K0] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
    2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    0.0f, 2.0f, 3.0f, 4.0f, 5.0f
 };
float h_b0[K0*N0] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 
    2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
    0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
    2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
    0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
   };

    float h_c0[M0*N0];
      
    float *d_a0, *d_b0, *d_c0;
    cudaMalloc((void**)&d_a0, M0*K0*sizeof(float));
    cudaMalloc((void**)&d_b0, K0*N0*sizeof(float));
    cudaMalloc((void**)&d_c0, M0*N0*sizeof(float));

    dimvec.push_back(std::make_tuple(M0, N0, K0));
    array_ptr_vec_host.push_back(std::make_tuple((float*)h_a0, (float*)h_b0, (float*)h_c0));
    array_ptr_vec_device.push_back(std::make_tuple((float*)d_a0, (float*)d_b0, (float*)d_c0));

    //initializing 1st GEMM operation
    size_t M1 = 3;
    size_t N1 = 7;
    size_t K1 = 3;
    
    float h_a1[M1*K1] = {1.0f, 0.0f, 0.0f, 
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f
                    };
    float h_b1[K1*N1] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 
                       2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
                       0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                      };
  
    float h_c1[M1*N1];
      
    float *d_a1, *d_b1, *d_c1;
    cudaMalloc((void**)&d_a1, M1*K1*sizeof(float));
    cudaMalloc((void**)&d_b1, K1*N1*sizeof(float));
    cudaMalloc((void**)&d_c1, M1*N1*sizeof(float));
    dimvec.push_back(std::make_tuple(M1, N1, K1));
    array_ptr_vec_host.push_back(std::make_tuple((float*)h_a1, (float*)h_b1, (float*)h_c1));
    array_ptr_vec_device.push_back(std::make_tuple((float*)d_a1, (float*)d_b1, (float*)d_c1));


    //initializing 2nd GEMM operation
    size_t M2 = 3;
    size_t N2 = 7;
    size_t K2 = 5;
    
    float h_a2[M2*K2] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
                       2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                       0.0f, 2.0f, 3.0f, 4.0f, 5.0f
                    };
    float h_b2[K2*N2] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 
                       2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
                       0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                       2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
                       0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
                      };
  
    float h_c2[M2*N2];
      
    float *d_a2, *d_b2, *d_c2;
    cudaMalloc((void**)&d_a2, M2*K2*sizeof(float));
    cudaMalloc((void**)&d_b2, K2*N2*sizeof(float));
    cudaMalloc((void**)&d_c2, M2*N2*sizeof(float));

    dimvec.push_back(std::make_tuple(M2, N2, K2));
    array_ptr_vec_host.push_back(std::make_tuple((float*)h_a2, (float*)h_b2, (float*)h_c2));
    array_ptr_vec_device.push_back(std::make_tuple((float*)d_a2, (float*)d_b2, (float*)d_c2));



     //initializing 3rd GEMM operation
     size_t M3 = 3;
     size_t N3 = 7;
     size_t K3 = 3;
     
     float h_a3[M3*K3] = {1.0f, 0.0f, 0.0f, 
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f
                     };
     float h_b3[K3*N3] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 
                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 7.0f,
                        0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                       };
   
     float h_c3[M3*N3];
       
     float *d_a3, *d_b3, *d_c3;
     cudaMalloc((void**)&d_a3, M3*K3*sizeof(float));
     cudaMalloc((void**)&d_b3, K3*N3*sizeof(float));
     cudaMalloc((void**)&d_c3, M3*N3*sizeof(float));
     dimvec.push_back(std::make_tuple(M3, N3, K3));
     array_ptr_vec_host.push_back(std::make_tuple((float*)h_a3, (float*)h_b3, (float*)h_c3));
     array_ptr_vec_device.push_back(std::make_tuple((float*)d_a3, (float*)d_b3, (float*)d_c3));


     //Now we do CUDA streams processing
     cudaStream_t stream0, stream1, stream2;
     cudaStreamCreate(&stream0);
     cudaStreamCreate(&stream1);
     cudaStreamCreate(&stream2);

     for(size_t i = 0; i < dimvec.size(); i +=stream_step){
        size_t ind0 = i;
        size_t ind1 = i + 1;
        size_t ind2 = i + 2;

        float* A0, *B0, *C0, *A1, *B1, *C1, *A2, *B2, *C2;
        float* h_A0, *h_B0, *h_C0, *h_A1, *h_B1, *h_C1, *h_A2, *h_B2, *h_C2;
        size_t M0, N0, K0, M1, N1, K1, M2, N2, K2;

        //// copy data
        A0 = std::get<0>(array_ptr_vec_device[ind0]);
        B0 = std::get<1>(array_ptr_vec_device[ind0]);
        C0 = std::get<2>(array_ptr_vec_device[ind0]);
        h_A0 = std::get<0>(array_ptr_vec_host[ind0]);
        h_B0 = std::get<1>(array_ptr_vec_host[ind0]);
        h_C0 = std::get<2>(array_ptr_vec_host[ind0]);
        M0 = std::get<0>(dimvec[ind0]);
        N0 = std::get<1>(dimvec[ind0]);
        K0 = std::get<2>(dimvec[ind0]);


        cudaMemcpyAsync(A0, h_A0, M0*K0*sizeof(float),cudaMemcpyHostToDevice , stream0);
        cudaMemcpyAsync(B0, h_B0, N0*K0*sizeof(float),cudaMemcpyHostToDevice , stream0);
    
        
        if(ind1 < dimvec.size()){
            A1 = std::get<0>(array_ptr_vec_device[ind1]);
            B1 = std::get<1>(array_ptr_vec_device[ind1]);
            C1 = std::get<2>(array_ptr_vec_device[ind1]);
            h_A1 = std::get<0>(array_ptr_vec_host[ind1]);
            h_B1 = std::get<1>(array_ptr_vec_host[ind1]);
            h_C1 = std::get<2>(array_ptr_vec_host[ind1]);
            M1 = std::get<0>(dimvec[ind1]);
            N1 = std::get<1>(dimvec[ind1]);
            K1 = std::get<2>(dimvec[ind1]);
            cudaMemcpyAsync(A1, h_A1, M1*K1*sizeof(float),cudaMemcpyHostToDevice , stream1);
            cudaMemcpyAsync(B1, h_B1, N1*K1*sizeof(float),cudaMemcpyHostToDevice , stream1);
        
        }
        if(ind2 < dimvec.size()){
            A2 = std::get<0>(array_ptr_vec_device[ind2]);
            B2 = std::get<1>(array_ptr_vec_device[ind2]);
            C2 = std::get<2>(array_ptr_vec_device[ind2]);
            h_A2 = std::get<0>(array_ptr_vec_host[ind2]);
            h_B2 = std::get<1>(array_ptr_vec_host[ind2]);
            h_C2 = std::get<2>(array_ptr_vec_host[ind2]);
            M2 = std::get<0>(dimvec[ind2]);
            N2 = std::get<1>(dimvec[ind2]);
            K2 = std::get<2>(dimvec[ind2]);
            cudaMemcpyAsync(A2, h_A2, M2*K2*sizeof(float),cudaMemcpyHostToDevice , stream2);
            cudaMemcpyAsync(B2, h_B2, N2*K2*sizeof(float),cudaMemcpyHostToDevice , stream2);
        }
        

        //execute kernels
        dim3 dimGrid(ceil((1.0*N0)/GEMM_TILE_WIDTH),ceil((1.0*M0)/GEMM_TILE_WIDTH), 1);//AL: Be careful of x, y z dimenions! Ordering matters
        dim3 dimBlock(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
        tiled_GEMM<<<dimGrid, dimBlock, 0, stream0>>>(A0, B0, C0, M0, N0, K0);

        if(ind1 < dimvec.size()){
            dim3 dimGrid(ceil((1.0*N1)/GEMM_TILE_WIDTH),ceil((1.0*M1)/GEMM_TILE_WIDTH), 1);//AL: Be careful of x, y z dimenions! Ordering matters
            dim3 dimBlock(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
            tiled_GEMM<<<dimGrid, dimBlock, 0, stream1>>>(A1, B1, C1, M1, N1, K1);
        }

        if(ind2 < dimvec.size()){
            dim3 dimGrid(ceil((1.0*N2)/GEMM_TILE_WIDTH),ceil((1.0*M2)/GEMM_TILE_WIDTH), 1);//AL: Be careful of x, y z dimenions! Ordering matters
            dim3 dimBlock(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
            tiled_GEMM<<<dimGrid, dimBlock, 0, stream2>>>(A2, B2, C2, M2, N2, K2);
        }


        //copying output data
        cudaMemcpyAsync(h_C0, C0, M0*N0*sizeof(float),cudaMemcpyDeviceToHost , stream0);
        cudaFree(A0);
        cudaFree(B0);
        cudaFree(C0);
       

        if(ind1 < dimvec.size()){
            cudaMemcpyAsync(h_C1, C1, M1*N1*sizeof(float),cudaMemcpyDeviceToHost , stream1);
            cudaFree(A1);
            cudaFree(B1);
            cudaFree(C1);
        }

        if(ind2 < dimvec.size()){
            cudaMemcpyAsync(h_C2, C2, M2*N2*sizeof(float),cudaMemcpyDeviceToHost , stream2);
            cudaFree(A2);
            cudaFree(B2);
            cudaFree(C2);
        }

   
     }
    
    
     for(size_t i = 0; i < dimvec.size(); ++i){
        print_matrix(std::get<2>(array_ptr_vec_host[i]), std::get<0>(dimvec[i]), std::get<1>(dimvec[i]));
     }

     return;

}