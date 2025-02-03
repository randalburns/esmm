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

// same as sequential, but not square
__global__ void esmm_sequential_ns (int rows, int columns, int inners, 
					int rblksz, int cblksz, 
					const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    const int row = blockIdx.x * rblksz + (threadIdx.x / cblksz);
    const int col = blockIdx.y * cblksz + (threadIdx.x % cblksz);

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
    return;
}

// multiple elements updated per thread
//
//   for blksz^2 in C with blksz threads
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

    int MT = blocksize;

    // RBTODO need to make dynamic
    float tmpres[2] = {0.0}; // thread results
    //float tmpres[MT] = {0.0}; // thread results

    // for a block of A and B
    for (int inner=0; inner < inners; inner += blocksize)
    {
        // each thread loads MT elements
	for (int dotidx=0; dotidx<MT; dotidx++)
	{
	  // Load lock of A and B into shared memory
          sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
          sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
	}

	
	/*
        // Check the context of sA and sB looks good
	for (int dotidx=0; dotidx<MT; dotidx++)
	{
	  // Load lock of A and B into shared memory
          //C[(row + dotidx) * columns + col]  = sB[dotidx * blocksize + coloff]; 
          C[(row + dotidx) * columns + col]  = sB[dotidx * blocksize + coloff]; 
	}
	__syncthreads();
	return;
	*/

	// Let's try the normal loop
        for (int i=0; i < blocksize; ++i)
        {
            for (int dotidx=0; dotidx < blocksize; dotidx++)
 	    {
               C[(row + dotidx) * columns + col] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff]; 
	    }
	}
        __syncthreads();
    }
    return;

    {
	// outer loop is offsets in C
	for (int dotidx=0; dotidx<MT; dotidx++)
	{
	    float Btmp = sB[dotidx * blocksize + coloff];
	    // inner loops is inners.  resuse Bvalue in register
            for (int i=0; i < 1; ++i)
            //for (int i=0; i < blocksize; ++i)
            {
		C[(row + dotidx) * blocksize + coloff] = sA[dotidx * blocksize + i] + Btmp;
                //tmpres[i] += sA[dotidx * blocksize + i] * Btmp;  
	    }
	}
        __syncthreads();
	return;

	// apply all updates to C
        for (int i=0; i < blocksize; ++i)
        {
          C[row * columns + col] = tmpres[i];
        }
        __syncthreads();
    }
}

//    C[row * columns + col] = 100 * blockIdx.x + 10* blockIdx.y + row + 0.1*col;
//    C[row * columns + col] = 100 * blockIdx.x + 10* blockIdx.y + rowoff + 0.1*coloff;
//    C[row * columns + col] = B[0];
