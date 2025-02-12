#include "esmm_gpu.h"
#include "esmm_cpu.h"

#include <iostream>
#include <chrono>


#define TIME_BLOCK_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_RESTART start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_END  { \
    auto stop = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl; \
}


int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 512;
    constexpr int columns = 512;
    constexpr int inners = 512; 
     
    dim3 gridDim(16,16);
    dim3 blockDim(32,32);

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

    float A[rows * inners];
    float B[inners * columns];
    float C[rows * columns];
    
    // Initialize A to random floating-point values between 0 and 1
    for (int row = 0; row < rows; ++row) {
        for (int inner = 0; inner < inners; ++inner) {
            A[row * inners + inner] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Initialize B to random floating-point values between 0 and 1
    for (int inner = 0; inner < inners; ++inner) {
        for (int col = 0; col < columns; ++col) {
            B[inner * columns + col] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
	    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice);

    // Tiled naive
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_START
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled naive kernel." << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled sequential
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Sequential kernel " << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled shared memory
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_sequential_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y * sizeof(float)>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled shared memory kernel " << std::endl;
    cudaMemset(d_C, 0, Csize);

    // multi
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_shmem_multi<<<gridDim, blockDim.x, 2 * blockDim.x * blockDim.y * sizeof(float)>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled multi " << std::endl;
    cudaMemset(d_C, 0, Csize);

    // multi 2
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_shmem_multi2<<<gridDim, blockDim.x, 2 * blockDim.x * blockDim.y * sizeof(float)>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled multi 2 " << std::endl;
    cudaMemset(d_C, 0, Csize);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
