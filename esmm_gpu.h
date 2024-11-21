#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void esmm_naive(int  rows, int columns, int inners, const float *A,
                           const float *B, float *C)
{
    // compute position in C that this thread is responsible for
    int ridx = blockIdx.x * blockDim.x + threadIdx.x;
    int cidx = blockIdx.y * blockDim.y + threadIdx.y;

    // along rows and colums of AB for each element in C 
    float tmp = 0.0;
    for (int i=0; i < inners; ++i)
    {
          tmp += A[ridx * inners + i] * B[i * columns + cidx]; 
    }
    C[ridx * columns + cidx] += tmp;
}

// only for square blocks
// launch 4x4 kernel as 16, blocksize = 4	
__global__ void esmm_sequential (int  rows, int columns, int inners, int blocksize, 
					const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    //   are the dims correct if not square?
    const int ridx = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int cidx = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    float tmp = 0.0;
    for (int i=0; i < inners; ++i)
    {
          tmp += A[ridx * inners + i] * B[i * columns + cidx]; 
    }
    C[ridx * columns + cidx] += tmp;
}

// blocks of B that access all row.  we will need to shrink at some point for shmem
__global__ void esmm_Btile (int rows, int columns, int inners, 
				int cTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    for (int row = 0; row < rows; row++)
    {
        for (int coloff = 0; coloff < cTileSize; coloff++)
        {
            int col = cTileOff + coloff;
	    int inner = iTileOff + threadIdx.x;

	    // RB atomic adds needed to prevent races.
            atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
	}
    }
}


// add shared memory to staggered version
__global__ void esmm_Btile_noatomic (int rows, int columns, int inners, 
				int cTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    for (int row = 0; row < rows; row++)
    {
        for (int coloff = 0; coloff < cTileSize; coloff++)
        {
            int col = cTileOff + (coloff + threadIdx.x) % cTileSize;
	    int inner = iTileOff + threadIdx.x;

	    // each kernel launch is fine.  multiple concurrent is not.
            //atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
            C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
	}
    }
}


__global__ void esmm_Btile_shmem (int  rows, int columns, int inners, 
				int cTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    // RB for now assume only one A tile.
    // Shared memory regions
    // A rTileSize x iTileSize
    // B iTileSize x cTileSize
    // C rTileSize x cTileSize
    
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;
    
    extern __shared__ float sArea [];
    float* sA = sArea;
    float* sB = sArea + iTileSize * rows;  // will have to tile rows later
    float* sC = sB + iTileSize * blockDim.z;

    // are there good shared memory loading patterns?
    
    // Load shared memory for A

    // Load shared memory for B. Each thread loads cTileSize elements of B
    for (int coloff = 0; coloff < cTileSize; coloff++)
    {
      int col = cTileOff + coloff;
      int inner = iTileOff + threadIdx.x;

      sB[threadIdx.x * cTileSize + coloff] = B[inner * columns + col];
    }

    // Load

    __syncthreads();

    // Check contents of shared memory
    for (int coloff = 0; coloff < cTileSize; coloff++)
    {
      C[threadIdx.x * cTileSize + coloff] = sB[threadIdx.x * cTileSize + coloff];
    }
}    


