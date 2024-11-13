#include <cuda_runtime.h>
#include <iostream>

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
