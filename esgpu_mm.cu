#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "esmm_cpu.h"


__global__ void esmm_nogrid (int rows, int inners, int columns, const float *A,
                           const float *B, float *C)
{
    int rTileOff = 0;
    int iTileOff = 0;
    int cTileOff = 0;

    int rTileSize = 4;
    int iTileSize = 4;

    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int row = rTileOff + rowoff;
            int inner = iTileOff + inneroff;

            // Unrolled coloff loop with columns == number threads
            C[row * columns + (cTileOff + threadIdx.x)] += A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.x)];

        }
    }
}

__global__ void esmm_tile (int rows, int inners, int columns, 
				int rTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    int rTileOff = blockIdx.x * rTileSize;
    int iTileOff = blockIdx.y * iTileSize;
    int cTileOff = blockIdx.z * blockDim.z; 

    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int row = rTileOff + rowoff;
            int inner = iTileOff + inneroff;

            // Unrolled coloff loop with columns == number threads
            //C[row * columns + (cTileOff + threadIdx.z)] += A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)];
            atomicAdd(&C[row * columns + (cTileOff + threadIdx.z)], A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)]);

        }
    }
}

__global__ void esmm_naive(int rows, int inners, int columns, const float *A,
                           const float *B, float *C)
{
    // compute position in C that this thread is responsible for
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // along rows and colums of AB for each element in C
    int i=0;
    float tmp = 0.0;
    for (; i < inners; ++i)
    {
	  // sums that work as a test/placeholder
          // Row of A
          //tmp += A[xidx * columns + i]; 
          // Col of B
          // tmp += B[i * inners + yidx]; 
	  // Row of A  Col of B Sum
          // tmp += A[xidx * columns + i] + B[i * inners + yidx]; 

	  // Multiply works on full grid
          tmp += A[xidx * columns + i] * B[i * inners + yidx]; 
    }
    C[xidx * columns + yidx] = tmp;
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

    // Zero target data
    cudaMemset(d_C, 0, Csize);

//    printf("\n A \n\n");
//    printMatrix<rows, columns>(A);
//    printf("\n B \n\n");
//    printMatrix<rows, columns>(B);


    // Launch kernel
    esmm_naive<<<gridDim, blockDim>>>(rows, inners, columns, d_A, d_B, d_C);

    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    printf("\n Base \n\n");
    printMatrix<rows, columns>(C);

    // Zero target matrix
    cudaMemset(d_C, 0, Csize);

    // Launch kernel
    //esmm_nogrid<<<1, 4>>>(rows, inners, columns, d_A, d_B, d_C);
    //esmm_tile<<<dim3(1,1,1), dim3(1,1,4)>>>(rows, inners, columns, 4, 4, d_A, d_B, d_C);
    //esmm_tile<<<dim3(2,2,1), dim3(1,1,4)>>>(rows, inners, columns, 2, 2, d_A, d_B, d_C);
    esmm_tile<<<dim3(4,4,1), dim3(1,1,4)>>>(rows, inners, columns, 1, 1, d_A, d_B, d_C);
    
    // Copy result from device to host
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);

    printf("\n Threaded \n\n");
    printMatrix<rows, columns>(C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
