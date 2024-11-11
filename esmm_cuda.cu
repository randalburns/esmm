#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "esmm_cpu.h"

// Kernel function for element-wise vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

// Kernel function for element-wise vector addition
__global__ void vectorAdd2d(const float *A, const float *B, float *C, int cols) {
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    C[xidx*cols+yidx] = A[xidx*cols+yidx] + B[xidx*cols+yidx];
}

__global__ void esmm_naive(int rows, int cols, int inners, const float *A,
                           const float *B, float *C)
{

    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    int i=0;

    float tmp = 0.0;
    for (; i < inners; ++i)
    {
        tmp += A[x * inners + i] * B[i * inners + y];
    }
    C[x * cols + y] += A[x * inners + i] * B[i * cols + y];
}

int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 4;
    constexpr int columns = 4;
    constexpr int inners = 4;

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 2.0, 3.0, 4.0,
                              2.0, 3.0, 4.0, 5.0,
                              3.0, 4.0, 5.0, 6.0,
                              4.0, 5.0, 6.0, 7.0};

    float B[inners * columns] = {1.0, 2.0, 3.0, 4.0,
                                  2.0, 3.0, 4.0, 5.0,
                                  3.0, 4.0, 5.0, 6.0,
                                  4.0, 5.0, 6.0, 7.0};

    float C[rows * columns];
    zeroMatrix<rows,columns>(C);
    
    dim3 gridDim(1,1,1);
    dim3 blockDim(4,4,1);

    // create as many blocks as necessary to map all of C
	// dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
	// 32 * 32 = 1024 thread per block
	// dim3 blockDim(32, 32, 1);
	
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice);

    printMatrix<rows, columns>(A);
    printMatrix<rows, columns>(B);

    // Launch kernel
    // vectorAdd<<<1, 16>>>(d_A, d_B, d_C);

    // Launch kernel
    vectorAdd2d<<<dim3(1,1,1), dim3(4,2,1)>>>(d_A, d_B, d_C, 4);

    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printMatrix<rows, columns>(C);

    return 0;

    // launch the asynchronous execution of the kernel on the device
	// The function call returns immediately on the host
	esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, A, B, C);

    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printMatrix<rows, columns>(C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}