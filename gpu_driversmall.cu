#include "esmm_gpu.h"
#include "esmm_cpu.h"

int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 2;
    constexpr int columns = 2;
    constexpr int inners = 2;

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 2.1,
                              2.0, 2.1};

    float B[inners * columns] = {1.0, 1.2,
                                 2.0, 2.2};
	    
    float C[rows * columns];
    
    dim3 gridDim(1,1);
    dim3 blockDim(2,2);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice);

    // Zero target data
    cudaMemset(d_C, 0, Csize);

    // Launch kernel
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);

    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Base \n\n");
    printMatrix<rows, columns>(C);

    // Sequential
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<dim3(1,1), dim3(2 * 2)>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // Sequential tiled
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<dim3(2,2), dim3(1 * 1)>>>(rows, columns, inners, 1, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential tiled -- 1x1\n\n");
    printMatrix<rows, columns>(C);

    // shared memory
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_sequential_shmem<<<dim3(1,1), dim3(2*2), 8>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // shared memory tiled
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_sequential_shmem<<<dim3(2,2), dim3(1*1), 2>>>(rows, columns, inners, 1, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory tiled -- 1x1 \n\n");
    printMatrix<rows, columns>(C);

    return;
}
