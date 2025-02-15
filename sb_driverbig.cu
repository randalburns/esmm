#include "esmm_cpu.h"
#include "sib_gpu.h"


#include <iostream>
#include <chrono>


#define TIME_BLOCK_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_RESTART start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_END  { \
    auto stop = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl; \
}

// Function to check if two matrices are equal within a tolerance
bool checkEqual(int rows, int cols, float* matrix1, float* matrix2, float tolerance = 1e-4) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (std::fabs(matrix1[row * cols + col] - matrix2[row * cols + col]) > tolerance) {
                std::cout << std::endl;
                std::cout << "Value1 " << matrix1[row * cols + col] 
                          << " Value2 " << matrix2[row * cols + col] 
               	          << " offset " << row << ", " << col 
                          << " Value2 " << matrix2[row * cols + col] 
			  << " Difference  "  
                          << std::fabs(matrix1[row * cols + col] - matrix2[row * cols + col]) << std::endl;
                return false;
            }
        }
    }
    return true;
}


int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 1024;
    constexpr int columns = 1024;
    constexpr int inners = 1024; 
     
    dim3 gridDim(32,32);
    dim3 blockDim(32,32);

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

    float* A = (float*)malloc(rows * inners * sizeof(float));
    float* B = (float*)malloc(inners * columns * sizeof(float));
    float* C = (float*)malloc(rows * columns * sizeof(float));
    float* Ccpu = (float*)malloc(rows * columns * sizeof(float));

    
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

    // baseMM on CPU
    zeroMatrix<rows,columns>(Ccpu);
    TIME_BLOCK_START
    baseMM<rows, columns, inners>(A, B, Ccpu);
    TIME_BLOCK_END
	    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice);

    // Tiled naive
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled naive kernel matches = " << checkEqual ( rows, columns, Ccpu, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled sequential
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Sequential kernel matches = " << checkEqual ( rows, columns, Ccpu, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y * sizeof(float)>>>
	    			(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled shared memory kernel = " << checkEqual ( rows, columns, Ccpu, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    sb_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y * sizeof(float)>>>
	    			(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "SB shared memory kernel = " << checkEqual ( rows, columns, Ccpu, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
