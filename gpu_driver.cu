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
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential \n\n");
    printMatrix<rows, columns>(C);

    // Sequential tiled
    cudaMemset(d_C, 0, Csize);
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential tiled \n\n");
    printMatrix<rows, columns>(C);

    // i tiled
    cudaMemset(d_C, 0, Csize);
    esmm_itile<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential I tiled \n\n");
    printMatrix<rows, columns>(C);

    // i tiled
    cudaMemset(d_C, 0, Csize);
    esmm_itile<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Sequential I tiled \n\n");
    printMatrix<rows, columns>(C);

    return;

    // Btile_Ithreads
    cudaMemset(d_C, 0, Csize);
    esmm_base<<<dim3(1,1), dim3(4)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n esmm base\n\n");
    printMatrix<rows, columns>(C);

    // Btile_Ithreads_tiled
    cudaMemset(d_C, 0, Csize);
    esmm_base<<<dim3(2,2), dim3(2)>>>(rows, columns, inners, 2, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n esmm base tiled \n\n");
    printMatrix<rows, columns>(C);

    //  decluster writes
    cudaMemset(d_C, 0, Csize);
    esmm_base_noatomic<<<dim3(1,1), dim3(4)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n esmm base tiled noatomic \n\n");
    printMatrix<rows, columns>(C);

    // shared memory
    zeroMatrix<rows,columns>(C);
    cudaMemset(d_C, 0, Csize);
    //esmm_shmem<<<dim3(1,1), dim3(4), 2*4*4*sizeof(float)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
    esmm_shmem<<<dim3(2,2), dim3(2), 2*4*4*sizeof(float)>>>(rows, columns, inners, 2, 2, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n esmm shmem \n\n");
    printMatrix<rows, columns>(C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/*
    // Launch kernel
    //esmm_Btile<<<dim3(1,1), dim3(4)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
    //esmm_Btile<<<dim3(2,2), dim3(2)>>>(rows, columns, inners, 2, 2, d_A, d_B, d_C);
    
   esmm_Btile_noatomic<<<dim3(1,1), dim3(4)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
//    esmm_Btile_noatomic<<<dim3(2,2), dim3(2)>>>(rows, columns, inners, 2, 2, d_A, d_B, d_C);
    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printf("\n Tiled \n\n");
    printMatrix<rows, columns>(C);

    // Zero target matrix
    cudaMemset(d_C, 0, Csize);

    // Launch kernel
    esmm_Btile_shmem<<<dim3(1,1), dim3(4), 3*4*4*sizeof(float)>>>(rows, columns, inners, 4, 4, d_A, d_B, d_C);
    
    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printf("\n Shmem \n\n");
    printMatrix<rows, columns>(C);

 
    // Zero target matrix
    cudaMemset(d_C, 0, Csize);

    //esmm_tile_noatomic<<<dim3(1,1,1), dim3(1,1,4)>>>(rows, inners, columns, 4, 4, d_A, d_B, d_C);
    esmm_tile_noatomic<<<dim3(2,2,1), dim3(1,1,4)>>>(rows, inners, columns, 2, 2, d_A, d_B, d_C);

    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printf("\n Tiled no atomic \n\n");
    printMatrix<rows, columns>(C);

*/

