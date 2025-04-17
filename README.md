This is a respository of implementations of certain CUDA parallel design patterns, including GEMM, Convolutions, Sparse Matrices, Parallel Scanning, and Histogramming.

To build with make on linux, run this command in the build directory:

cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER=nvcc ..
