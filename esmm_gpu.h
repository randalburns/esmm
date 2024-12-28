#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void esmm_naive(int rows, int columns, int inners, const float *A,
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
// grid still 2d
// launch 4x4 kernel as 16, blocksize = 4	
__global__ void esmm_sequential (int  rows, int columns, int inners, int blocksize, 
					const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    const int row = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    float tmp = 0.0;
    for (int i=0; i < inners; ++i)
    {
          tmp += A[row * inners + i] * B[i * columns + col]; 
    }
    C[row * columns + col] += tmp;
}

// each thread is an output element of C
__global__ void esmm_itile (int rows, int columns, int inners,
                           int blocksize, int iTileSize,
                           const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    const int row = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    for (int itile=0; itile < inners / iTileSize; itile++)
    {
        float tmp = 0.0;
        for (int i=0; i < iTileSize; i++)
        {
            int inneroff = itile * iTileSize + i;
            tmp += A[row * inners + inneroff] * B[inneroff * columns + col]; 
        }
        C[row * columns + col] += tmp;
    }

/*    // change iteration order to output sequentially
    //   are the dims correct if not square?
    const int row = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    float tmp = 0.0;
    for (int i=0; i < inners; ++i)
    {
          tmp += A[ridx * inners + i] * B[i * columns + cidx]; 
    }
    C[ridx * columns + cidx] += tmp;
    */
}

// Each thread is a column of inners in B.  Need to tile Rows
__global__ void esmm_base (int rows, int columns, int inners,
                           int cTileSize, int iTileSize,
                           const float *A, const float *B, float *C)
{
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    for (int row = 0; row < rows; row++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int col = cTileOff + threadIdx.x;
            int inner = iTileOff + inneroff;

            // RB atomic adds needed to prevent races.
            atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
            //C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
        }
    }
}

// decluster writes so that they don't conflict
// this does not work when multiple kernels update the same data -- what about launching 32 threads at a time?
__global__ void esmm_base_noatomic (int rows, int columns, int inners,
                           int cTileSize, int iTileSize,
                           const float *A, const float *B, float *C)
{
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    for (int row = 0; row < rows; row++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int col = cTileOff + threadIdx.x;
            int inner = iTileOff + (inneroff + threadIdx.x) % iTileSize;

            // RB atomic adds needed to prevent races.
            //atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
            C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
        }
    }
}


// shared memory 
__global__ void esmm_shmem (int rows, int columns, int inners,
                           int cTileSize, int iTileSize,
                           const float *A, const float *B, float *C)
{
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    // RB for now assume only one A tile.
    // Shared memory regions
    // A rTileSize x iTileSize
    // B iTileSize x cTileSize
    // C rTileSize x cTileSize
    
    extern __shared__ float sArea [];
    float* sB = sArea;  
    float* sC = sB + cTileSize * blockDim.x;

    // Load shared memory for B. Each thread loads iTileSize elements of B
    //  each thread is loading a col B
    for (int inneroff = 0; inneroff < iTileSize; inneroff++)
    {
      int col = cTileOff + threadIdx.x;
      int inner = iTileOff + inneroff;

      sB[inneroff * cTileSize + threadIdx.x] = B[inner * columns + col];
    }

    __syncthreads();

    for (int row = 0; row < rows; row++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int col = cTileOff + threadIdx.x;
            int inner = iTileOff + inneroff;

            // RB atomic adds needed to prevent races.
	    // use sC
            // atomicAdd(&sC[row * cTileSize + threadIdx.x], A[row * inners + inner] * sB[inneroff * cTileSize + threadIdx.x]);
            atomicAdd(&C[row * columns + col], A[row * inners + inner] * sB[inneroff * cTileSize + threadIdx.x]);

            //C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
        }
    }

    return;
    
    __syncthreads();


    // copy sC to C
    for (int row = 0; row < rows; row++)
    {
        for (int coloff = 0; coloff < cTileSize; coloff++)
        {
          int col = cTileOff + threadIdx.x;
          C[row * columns + col] = sC[row * cTileSize + threadIdx.x];
	}
    }

    // dumb check of shared memory
//    for (int row = 0; row < rows; row++) {
//      for (int col = 0; col < columns; col++) {
//	      C[row * columns + col] = float(row * columns + col);
//      }
//    }

    // Check contents of shared memory B
//    for (int inneroff = 0; inneroff < iTileSize; inneroff++)
//    {
//      C[inneroff * iTileSize + threadIdx.x] = sB[inneroff * iTileSize + threadIdx.x];
//    } 
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
    float* sC = sB + iTileSize * blockDim.x;

    // are there good shared memory loading patterns?
    //   test thread continguous versus thread loop
    
    // Load shared memory for A
    //   this assumes that a full warp is being load, no check on
    //      if (threadIdx.x < iTileSize)
    //
    // Each thread is loading a columns of inners, i.e. row x inners in A
    for (int rowoff=0; rowoff < rows; rowoff++)
    {
        sA[rowoff * iTileSize + threadIdx.x] = A[rowoff * inners + iTileOff + threadIdx.x];
    }

    // Load shared memory for B. Each thread loads cTileSize elements of B
    //  each thread is loading a row of inners in B, i.e. inners x cols in B
    for (int coloff = 0; coloff < cTileSize; coloff++)
    {
      int col = cTileOff + coloff;
      int inner = iTileOff + threadIdx.x;

      sB[threadIdx.x * cTileSize + coloff] = B[inner * columns + col];
    }

    // Load

    __syncthreads();

    // Check contents of shared memory A
    /*for (int rowoff = 0; rowoff < rows; rowoff++)
    {
      C[rowoff * iTileSize + threadIdx.x] = sA[rowoff * iTileSize + threadIdx.x];
    }*/

    // Check contents of shared memory B
    /* for (int coloff = 0; coloff < cTileSize; coloff++)
    {
      C[threadIdx.x * cTileSize + coloff] = sB[threadIdx.x * cTileSize + coloff];
    } */



}    


