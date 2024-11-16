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
  //          C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
	}
    }
}

// blocks of B that access all row.  we will need to shrink at some point for shmem
//   decluster writes so not atomic
__global__ void esmm_Btile_noatomic (int rows, int columns, int inners, 
				int cTileSize, int iTileSize, 
				const float *A, const float *B, float *C)
{
    // Shared memory regions
    // A rows x iTileSize // need to block rows.
    // B iTileSize x cTileSize
    // C rTileSize x cTileSize
    
    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;
    
    extern __shared__ float sArea [];
    float sA = sArea;
    float sB = sArea + iTileSize x rows;  // will have to tile rows later
    float sC = sB + iTilesize * blockDim.x;

    int iTileOff = blockIdx.x * iTileSize;
    int cTileOff = blockIdx.y * cTileSize;

    // Load shared memory.  

    // each thread loads cTileSize elements of B
    for (int coloff = 0; coloff < cTileSize; coloff++)
    {
      sB{(threadIdx.x * iTileSize + coloff) = B[(rTileOff + threadIdx.x) * cols + cTileOff + coloff];
    }

    // Each thread loads rows elements of A
    for (int rowoff = 0; rowoff < rTileSize; rowoff++)
    {
      sA[idxThread.x + rowoff * iTileSize] = 
	      A[(rTileOff + idxThread.x) + inners + (rTileOff + rowoff)];
    }

    // RB TEST to here.
    return 0;

    for (int row = 0; row < rows; row++)
    {
        for (int coloff = 0; coloff < cTileSize; coloff++)
        {
            int col = cTileOff + (coloff + threadIdx.x) % cTileSize;
	    int inner = iTileOff + threadIdx.x;

	    // each kernel launch is fine.  multiple concurrent is not.
            atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
           //C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
	}
//        __syncthreads () ;
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
            atomicAdd(&C[row * columns + col], A[row * inners + inner] * B[inner * columns + col]);
           //C[row * columns + col] += A[row * inners + inner] * B[inner * columns + col];
	}
//        __syncthreads () ;
    }
}
/*
__global__ void esmm_tile_no_atomic_shmem (int  rows, int columns, int inners, 
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
    float sA = sArea;
    float sB = sArea + iTileSize x rows;  // will have to tile rows  
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
