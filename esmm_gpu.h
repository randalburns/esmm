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


// tiled so that each thread (inners) computers all columns??
__global__ void esmm_tile (int  rows, int columns, int inners, 
				int rTileSize, int cTileSize, 
				const float *A, const float *B, float *C)
{
    int rTileOff = blockIdx.x * rTileSize;
    int cTileOff = blockIdx.y * cTileSize; 
    int iTileOff = blockIdx.z * blockDim.x;

    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
        for (int coloff = 0; coloff < cTileSize; coloff++)
        {
            int row = rTileOff + rowoff;
            int col = cTileOff + coloff;
	    int inner = iTileOff + threadIdx.x;

            atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + cTileOff + col]);

        }
    }
}

/*

// RB I think that this is tiled wrong. try again.
// tiled for thread per row in B 
__global__ void esmm_tile (int  rows, int columns, int inners, 
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

            atomicAdd(&C[row * columns + (cTileOff + threadIdx.z)], A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)]);

        }
    }
}

// Stagger the I/O by column so that there are no conflicting writes.
__global__ void esmm_tile_noatomic (int  rows, int columns, int inners, 
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
            int row = rTileOff + (rowoff + blockDim.z) % rTileSize;
            int inner = iTileOff + inneroff; 

	    // RB OK this works for 1 block. but different blocks are racy. SHMEM to the rescue?
            atomicAdd(&C[row * columns + (cTileOff + threadIdx.z)], A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)]);

//            C[row * columns + (cTileOff + threadIdx.z)] += A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)];
        }
    }
}

/*
// Stagger the I/O by column so that there are no conflicting writes.
__global__ void esmm_tile_no_atomic_shmem (int  rows, int columns, int inners, 
				int rTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    // Shared memory regions
    // A iTileSize x rTileSize
    // B iTileSize x cTileSize
    // C rTileSize x cTileSize
    
    extern __shared__ float sArea [];
    float sA = sArea;
    float sB = sArea + iTileSize x rTileSize;
    float sC = sB + iTilesize * blockDim.z;

    // Load shared memory.  We have cTileSize threads

    // each thread loads cTileSize elements of B
    for (int inneroff = 0; inneroff < iTileSize; inneroff++)
    {
      int inner = iTileOff + inneroff; 
      sB[threadIdx.z * cTileSize + inneroff] = B[inner * columns + (cTileOff + threadIdx.z)]
    }

    // each thread loads row rTileSize of A
    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
      sA
    }

    

    
    int rTileOff = blockIdx.x * rTileSize;
    int iTileOff = blockIdx.y * iTileSize;
    int cTileOff = blockIdx.z * blockDim.z; 

    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
        for (int inneroff = 0; inneroff < iTileSize; inneroff++)
        {
            int row = rTileOff + (rowoff + blockDim.z) % rTileSize;
            int inner = iTileOff + inneroff; 

	    // RB OK this works for 1 block. but different blocks are racy. SHMEM to the rescue?
            atomicAdd(&C[row * columns + (cTileOff + threadIdx.z)], A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)]);

//            C[row * columns + (cTileOff + threadIdx.z)] += A[row * inners + inner] * B[inner * columns + (cTileOff + threadIdx.z)];
        }
    }
}
*/
