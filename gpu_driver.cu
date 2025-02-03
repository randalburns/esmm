#include "esmm_gpu.h"
#include "esmm_cpu.h"

int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 4;
    constexpr int columns = 4;
    constexpr int inners = 4;

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 1.1, 1.2, 1.3,
                              2.0, 2.1, 2.2, 2.3,
                              3.0, 3.1, 3.2, 3.3,
                              4.0, 4.1, 4.2, 4.3};

    float B[inners * columns] = {1.0, 1.2, 1.4, 1.6,
                              2.0, 2.2, 2.4, 2.6,
                              3.0, 3.2, 3.4, 3.6,
                              4.0, 4.2, 4.4, 4.6};
	    
    float C[rows * columns];
    
    dim3 gridDim(1,1);
    dim3 blockDim(4,4);

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
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<dim3(1,1), 4 * 4>>>(rows, columns, inners, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // Sequential tiled
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<dim3(2,2), 2*2>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential -- 2x2 \n\n");
    printMatrix<rows, columns>(C);

    // shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_sequential_shmem<<<dim3(1,1), 4*4, 4*4*2>>>(rows, columns, inners, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // shared memory tiled
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_sequential_shmem<<<dim3(2,2), 2*2, 2*2*2>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential shared memory -- 2x2 \n\n");
    printMatrix<rows, columns>(C);

    // multi 
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_shmem_multi<<<dim3(1,1), 4, 4*4*2>>>(rows, columns, inners, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // multi tiled
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_shmem_multi<<<dim3(2,2), 2, 2*2*2>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi tiled -- 2x2 \n\n");
    printMatrix<rows, columns>(C);

    // multi 
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_shmem_multi2<<<dim3(1,1), 4, 4*4*2>>>(rows, columns, inners, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi 2 -- 4x4 \n\n");
    printMatrix<rows, columns>(C);

    // multi2 tiled
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    esmm_shmem_multi2<<<dim3(2,2), 2, 2*2*2>>>(rows, columns, inners, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Multi 2 tiled -- 2x2 \n\n");
    printMatrix<rows, columns>(C);

    return;
}
