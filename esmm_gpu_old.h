
/*

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
}

/*
// shared memory 
__global__ void esmm_shmem (int rows, int columns, int inners,
                           int blocksize, int iTileSize,
                           const float *A, const float *B, float *C)
{
    // change iteration order to output sequentially
    const int row = blockIdx.x * blocksize + (threadIdx.x / blocksize);
    const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);
    

    // RB for now assume only one A tile.
    // Shared memory regions
    // A rTileSize x iTileSize
    // B iTileSize x cTileSize
    // C rTileSize x cTileSize
    
    extern __shared__ float sArea [];
    float* sB = sArea;  
    float* sC = sB + blocksize * blockDim.x;

    // Load shared memory for B. Each thread loads iTileSize elements of B
    //  each thread is loading a col B
    for (int inneroff = 0; inneroff < iTileSize; inneroff++)
    {
      int col = cTileOff + threadIdx.x;
      int inner = iTileOff + inneroff;
