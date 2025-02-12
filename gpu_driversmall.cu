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
    float A[rows * inners] = {1.0, 2.0,
                              3.0, 4.0};

    float B[inners * columns] = {0.1, 0.2,
                                 0.3, 0.4};
	    
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

    printf("\n A \n\n");
    printMatrix<rows, columns>(A);

    printf("\n B \n\n");
    printMatrix<rows, columns>(B);

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
    esmm_sequential_shmem<<<dim3(1,1), dim3(2*2), 2*2*2*sizeof(float)>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // shared memory tiled
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_sequential_shmem<<<dim3(2,2), dim3(1*1), 2*sizeof(float)>>>(rows, columns, inners, 1, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory tiled -- 1x1 \n\n");
    printMatrix<rows, columns>(C);

    // shmem_multi
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_shmem_multi<<<dim3(1,1), dim3(2), 2*2*2*sizeof(float)>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi \n\n");
    printMatrix<rows, columns>(C);

    // shmem_multi tiled
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_shmem_multi<<<dim3(2,2), dim3(1), 2*sizeof(float)>>>(rows, columns, inners, 1, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi tiled -- 1x1 \n\n");
    printMatrix<rows, columns>(C);

    // shmem_multi2
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_shmem_multi2<<<dim3(1,1), dim3(2), 2*2*2*sizeof(float)>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi 2 \n\n");
    printMatrix<rows, columns>(C);

    // shmem_multi2 tiled
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<2,2>(C);
    esmm_shmem_multi2<<<dim3(2,2), dim3(1), 2*sizeof(float)>>>(rows, columns, inners, 1, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi 2 tiled -- 1x1 \n\n");
    printMatrix<rows, columns>(C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
