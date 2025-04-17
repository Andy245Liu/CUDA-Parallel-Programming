#include "vector_add.cuh"
#include <iostream>
#include <cuda_runtime.h>


__global__ void add_vectors(const float *a, const float *b, float *c, const int N) {
    int idx = threadIdx.x; // Use thread index as position
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}



void wrap_test_vector_add() {
  printf("calling CUDA Wrapper!\n");
  const int N = 3;
  float h_a[N] = {1.0f, 2.0f, 3.0f};
  float h_b[N] = {3.0f, 4.0f, 5.0f};
  float h_c[N];

  // Device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, N * sizeof(float));
  cudaMalloc((void**)&d_b, N * sizeof(float));
  cudaMalloc((void**)&d_c, N * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch CUDA kernel (1 block, 2 threads)
  add_vectors<<<1, N>>>(d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Print results
  std::cout << "Result of Vector Add: ["; 
  for(int i = 0; i < N; ++i){
    std::cout << h_c[i];
    if(i < N-1){
      std::cout<<" ";
    }
    else{
      std::cout<<"]\n";
    }
  }
  

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return;
}