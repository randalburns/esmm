#ifndef ESMMGPU_H
#define ESMMGPU_H

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


// 2-d blocks. 1 element per thread
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
        sA[rowoff * blocksize + coloff] = A[row * inners + inner + coloff];
        sB[rowoff * blocksize + coloff] = B[(inner + rowoff) * columns + col];
	__syncthreads();

        for (int i=0; i < blocksize; ++i)
        {
            tmp += sA[rowoff * blocksize + i] * sB[i * blocksize + coloff]; 
	}
        __syncthreads();
    }

    C[row * columns + col] = tmp;
// Check memory contents.  for debugging.
//    C[row * columns + col] = B[row * columns + col];
    return;
}

// multiple elements updated per thread
//
//   for blksz^2 in C with blksz threads
//
//   multi -- inners outerloop, blocksize innerloop
//     this allows us to dot he register reuse of Btmp
__global__ void esmm_shmem_multi (int rows, int columns, int inners, 
			   	int blocksize,
		       	        const float *A, const float *B, float *C)
{
    // 1-d array of threads
    const int row = blockIdx.x * blocksize;
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    int coloff = col % blocksize;

    extern __shared__ float sArea [];
    float* sA = sArea;  
    float* sB = sArea + blocksize * blocksize; 

    // RBTODO need to make dynamic
    float tmpres[32] = {0.0}; // thread results

    // for a block of A and B
    for (int inner=0; inner < inners; inner += blocksize)
    {
        // each thread loads MT elements
	for (int dotidx=0; dotidx<blocksize; dotidx++)
	{
	  // Load lock of A and B into shared memory
          sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
          sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
	}
        __syncthreads();

	// This loop order not the same as siebohm
	//  over all the inners
        for (int i=0; i < blocksize; ++i)
        {
            // reuse a single element of B and apply to all thread partials
	    float Btmp = sB[i * blocksize + coloff];
            for (int dotidx=0; dotidx < blocksize; dotidx++)
 	    {
               tmpres[dotidx] +=  sA[dotidx * blocksize + i] * Btmp;
	    }
	}
        __syncthreads();
    }

    for (int dotidx=0; dotidx<blocksize; dotidx++)
    {
        C[(row + dotidx) * columns + col] = tmpres[dotidx];
    }
    return;
}

// multiple elements updated per thread
//
//   for blksz^2 in C with blksz threads
//     we can do the inner sweep on a smaller size
//     
// mutlti2 inners outerloop
__global__ void esmm_shmem_multi2 (int rows, int columns, int inners, 
			   	int blocksize,
		       	        const float *A, const float *B, float *C)
{
    // 1-d array of threads
    const int row = blockIdx.x * blocksize;
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

    int coloff = col % blocksize;

    extern __shared__ float sArea [];
    float* sA = sArea;  
    float* sB = sArea + blocksize * blocksize; 

    // RBTODO need to make dynamic
    float tmpres[32] = {0.0}; // thread results

    // for a block of A and B
    for (int inner=0; inner < inners; inner += blocksize)
    {
        // each thread loads MT elements
	for (int dotidx=0; dotidx<blocksize; dotidx++)
	{
	  // Load lock of A and B into shared memory
          sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
          sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
	}
        __syncthreads();

        for (int dotidx=0; dotidx < blocksize; dotidx++)
        {
            for (int i=0; i < blocksize; ++i)
 	    {
               tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
	    }
	}
        __syncthreads();
    }

    // each thread loads blocksize elements
    for (int dotidx=0; dotidx<blocksize; dotidx++)
    {
        C[(row + dotidx) * columns + col] = tmpres[dotidx];
    }
    return;
}

// this is an unrolled version for the future?
//   should serve as inner loop for multi2
__device__ void multiply_dense8 (int dotidx, int i, int blocksize, int coloff, float* tmpres, float* sA, float * sB)
{
               tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+1] * sB[(i+1) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+2] * sB[(i+2) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+3] * sB[(i+3) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+4] * sB[(i+4) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+5] * sB[(i+5) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+6] * sB[(i+6) * blocksize + coloff];
               tmpres[dotidx] +=  sA[dotidx * blocksize + i+7] * sB[(i+7) * blocksize + coloff];
               return;
}


#endif
