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
__global__ void esmm_sequential (int rows, int columns, int inners, int blocksize, 
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
    C[row * columns + col] = tmp;
}

__global__ void esmm_sequential_shmem (int rows, int columns, int inners, int blocksize, 
					const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    const int row = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    int rowoff = row % blocksize;
    int coloff = col % blocksize;

    extern __shared__ float sArea [];
    float* sA = sArea;  
    float* sB = sArea + blocksize * blocksize; 

    float tmp = 0.0;

    // for a block of A and B
    for (int inner=0; inner < inners; inner += blocksize)
    {
	// Load block of A and B into shared memory
        sA[rowoff * blocksize + coloff] = A[row * inners + inner + col];
        sB[rowoff * blocksize + coloff] = B[(inner + rowoff) * columns + col];
//        C[row * columns + col] = B[(inner + rowoff) * columns + col];
        C[row * columns + col] = coloff;
	__syncthreads();

// check shmem load
//        C[row * columns + coloff] = sB[rowoff * blocksize + coloff];
// check shmem load
//        C[row * columns + coloff] = sA[rowoff * blocksize + coloff];
//        C[row * columns + coloff] = sB[rowoff * blocksize + coloff];

        for (int i=0; i < blocksize; ++i)
        {
            tmp += sA[rowoff * blocksize + i] * sB[i * blocksize + coloff]; 
	}
        __syncthreads();
    }

//    C[row * columns + col] = tmp;
    return;
}

