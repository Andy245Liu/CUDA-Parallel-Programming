#include "sparse_matrix_multiply.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void SpMV_kernel( JDS jds, float *vec, float* output, int M){
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M){
      float dot = 0;
      unsigned int sec = 0;
  
      while(jds.col_ptr[sec+1] - jds.col_ptr[sec] > row){//check if this row has enough nonzero entries to warrant thread work
       
        dot += jds.data[jds.col_ptr[sec] + row] * vec[jds.col_index[jds.col_ptr[sec] + row]];
        sec++;
      }
      output[jds.row_perm[row]] = dot;
      // printf("RESULT %f %d\n", output[jds.row_perm[row]], jds.row_perm[row]);

    }
    
    return;
}


void wrap_test_SpMV(){

  std::cout<<"RUNNING SpMV test\n";


  //Preparing input sparse matrix
  int M = MATRIX_SIZE;
  int N = MATRIX_SIZE;
  float A[M * N]{
    3, 0, 1, 0,
    0, 0, 0, 0,
    0, 2, 4, 1,
    1, 0, 0, 1
  };

  CSR* csr = matrix_to_CSR(A, M, N);
  JDS* jds = CSR_to_transposed_JDS(csr, M, N);

  std::cout<<"Result of conversion to JDS:\n";
  std::cout<<"JDS data: ";
  for(int i = 0; i < jds->count; ++i){
    std::cout<<jds->data[i]<<" ";
  }
  std::cout<<'\n';
  std::cout<<"JDS col_index: ";
  for(int i = 0; i < jds->count; ++i){
    std::cout<<jds->col_index[i]<<" ";
  }
  std::cout<<'\n';
  std::cout<<"JDS row_perm: ";
  for(int i = 0 ; i < M; ++i){
    std::cout<<jds->row_perm[i]<<" ";
  }
  std::cout<<'\n';
  std::cout<<"JDS col_ptr: ";
  for(int i = 0 ; i <= jds->max_nonzeros; ++i){ 
    std::cout<<jds->col_ptr[i]<<" ";
  }
  std::cout<<'\n';


  //Preparing input vector

  float vec[N] = {1.0,2.0,3.0,4.0};

  //preparing output vector
  float output[M];



   // Device memory
   JDS d_jds;
   float* d_vec;
   float* d_output;
   cudaMalloc((void**)&d_jds, sizeof(JDS));
   cudaMalloc((void**)&(d_jds.data), jds->count * sizeof(float));
   cudaMalloc((void**)&(d_jds.col_index), jds->count * sizeof(int));
   cudaMalloc((void**)&(d_jds.col_ptr),  (jds->max_nonzeros+1)* sizeof(int));
   cudaMalloc((void**)&(d_jds.row_perm), M * sizeof(int));
   cudaMalloc((void**)&d_vec, N*sizeof(float));
   cudaMalloc((void**)&d_output, M*sizeof(float));



   // Copy input data to device
   cudaMemcpy(d_jds.data, jds->data,  jds->count * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_jds.col_index, jds->col_index,  jds->count * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_jds.col_ptr, jds->col_ptr,  (jds->max_nonzeros+1) * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_jds.row_perm, jds->row_perm,  M * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(&d_jds, jds,  sizeof(JDS), cudaMemcpyHostToDevice);
   cudaMemcpy(d_vec, vec,  N*sizeof(float), cudaMemcpyHostToDevice);






   //configuring kernel. Each thread operates on a row of the sparese matrix (i.e., output entry)
   dim3 dimGrid(ceil((1.0*M)/BLOCK_SIZE),1, 1);//AL: Be careful of x, y z dimenions! Ordering matters
   dim3 dimBlock(BLOCK_SIZE, 1, 1);

   // Launch CUDA kernel 
   SpMV_kernel<<<dimGrid, dimBlock>>>( d_jds, d_vec, d_output, M);
 
   cudaDeviceSynchronize();
   cudaMemcpy(output, d_output,  M * sizeof(float), cudaMemcpyDeviceToHost);
   std::cout << "Result of SpMV: [ ";

   for(int i = 0; i < M; ++i){
      std::cout<<output[i]<<" ";
   }
   std::cout<<"]\n";


  //Freeing CUDA memory

  cudaFree(d_jds.data);
  cudaFree(d_jds.col_index);
  cudaFree(d_jds.col_ptr);
  cudaFree(d_jds.row_perm);
  cudaFree(&d_jds);
  cudaFree(d_vec);
  cudaFree(d_output);

  //Freeing dynamically-allocated csr memory;
  free(csr->data);
  csr->data = NULL;
  free(csr->col_index);
  csr->col_index = NULL;
  free(csr->row_ptr);
  csr->row_ptr = NULL;
  free(csr);
  csr = NULL;

  free(jds->data);
  jds->data = NULL;
  free(jds->col_index);
  jds->col_index = NULL;
  free(jds->col_ptr);
  jds->col_ptr = NULL;
  free(jds->row_perm);
  jds->row_perm = NULL;
  free(jds);
  jds = NULL;
  return;
  }

