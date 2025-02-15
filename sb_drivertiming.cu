#include "esmm_cpu.h"
#include "sib_gpu.h"

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>



#define TIME_BLOCK_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_RESTART start = std::chrono::high_resolution_clock::now();
#define TIME_BLOCK_END  { \
    auto stop = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl; \
}

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)



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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 2048;
    constexpr int columns = 2048;
    constexpr int inners = 2048; 
     
    dim3 gridDim(64,64);
    dim3 blockDim(32,32);

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

    float* A = (float*)malloc(rows * inners * sizeof(float));
    float* B = (float*)malloc(inners * columns * sizeof(float));
    float* C = (float*)malloc(rows * columns * sizeof(float));
    float* Cref = (float*)malloc(rows * columns * sizeof(float));

    
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
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_START
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(Cref, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled naive kernel matches " << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled sequential
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Sequential kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    esmm_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y * sizeof(float)>>>
	    			(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled shared memory kernel = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // sb tiled shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    sb_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y * sizeof(float)>>>
	    			(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "SB shared memory kernel = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // sb 1-d
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    sb_1dwarp_tile<<<gridDim, blockDim.x * blockDim.y / TM, 2 * blockDim.x * blockDim.y / TM * sizeof(float)>>>
	    			(rows, columns, inners, 
				 blockDim.x, blockDim.y, blockDim.x / TM,
				 d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaCheckErrors("1-d tiling");
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "SB 1-d tiled = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // sb 1-d
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    TIME_BLOCK_RESTART
    sb_1dwarp_tile<<<dim3(32,32), 64 * 64 / TM, 2 * 64 * 64 / TM * sizeof(float)>>>
	    			(rows, columns, inners, 
				 64, 64, 64 / TM,
				 d_A, d_B, d_C);
    TIME_BLOCK_END
    cudaCheckErrors("1-d tiling (manual)");
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "SB 1-d tiled = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);


/*
    // CU blas
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;  // Scaling factor
    const float beta = 0.0f;   // No addition to existing C

    // Perform the operation C = alpha * A * B + beta * C
    // A (m × k), B (k × n), C (m × n)
    TIME_BLOCK_RESTART
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                rows, columns, inners,
                &alpha,
                d_A, rows,
                d_B, inners,
                &beta,
                d_C, rows);
    TIME_BLOCK_END
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "cuBlas = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);
    cublasDestroy(handle);
*/


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
