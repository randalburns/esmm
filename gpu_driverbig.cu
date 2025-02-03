#include "esmm_gpu.h"
#include "esmm_cpu.h"


// Function to check if two matrices are equal within a tolerance
bool checkEqual(int rows, int cols, float* matrix1, float* matrix2, float tolerance = 1e-6) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (std::fabs(matrix1[row * cols + col] - matrix2[row * cols + col]) > tolerance) {
                std::cout << std::endl;
                std::cout << "Value1 " << matrix1[row * cols + col] 
                          << " Value2 " << matrix2[row * cols + col] 
               	          << " offset " << row << ", " << col 
                          << " Value2 " << matrix2[row * cols + col] 
			  << " Difference  "  
                          << std::fabs(matrix1[row * cols + col] - matrix2[row * cols + col]) << std::endl;
                return false;
            }
        }
    }
    return true;
}


int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 32;
    constexpr int columns = 32;
    constexpr int inners = 32; 
     
    dim3 gridDim(2,2);
    dim3 blockDim(16,16);

    // rectangular 2,4
    dim3 gridDim24(2,4);
    dim3 blockDim24(16,8);
    
    // rectangular 4,2
    dim3 gridDim42(4,2);
    dim3 blockDim42(8,16);

    size_t Asize = rows * inners * sizeof(float);
    size_t Bsize = inners * columns * sizeof(float);
    size_t Csize = rows * columns * sizeof(float);

    float A[rows * inners];
    float B[inners * columns];
    float C[rows * columns];
    float Cref[rows * columns];
    float Ccpu[rows * columns];
    
    // Initialize A to random floating-point values between 0 and 1
    for (int row = 0; row < rows; ++row) {
        for (int inner = 0; inner < inners; ++inner) {
            A[row * inners + inner] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Initialize B to random floating-point values between 0 and 1
    for (int inner = 0; inner < inners; ++inner) {
        for (int col = 0; col < columns; ++col) {
            B[inner * columns + col] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // baseMM on CPU
    zeroMatrix<rows,columns>(Ccpu);
    baseMM<rows, columns, inners>(A, B, Ccpu);
	    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice);


    // Tiled naive
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(Cref);
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    cudaMemcpy(Cref, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled naive kernel matches = " << checkEqual ( rows, columns, Ccpu, Cref ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled sequential
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_sequential<<<gridDim, blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Sequential kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled not square	
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_sequential_ns<<<gridDim24, blockDim24.x * blockDim24.y>>>(rows, columns, inners, blockDim24.x, blockDim24.y, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Not square 24 kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled not square	
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_sequential_ns<<<gridDim42, blockDim42.x * blockDim42.y>>>(rows, columns, inners, blockDim42.x, blockDim42.y, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Not square 42 kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled shared memory
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_sequential_shmem<<<dim3(2,2), 16*16, 16*16*2>>>(rows, columns, inners, 16, d_A, d_B, d_C);
//    esmm_sequential_shmem<<<gridDim, blockDim.x * blockDim.y, 2 * blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled shared memory kernel = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // multi
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_shmem_multi<<<gridDim, blockDim.x, 2 * blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled multi matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // multi2
    cudaMemset(d_C, 0, Csize);
    zeroMatrix<rows,columns>(C);
    esmm_shmem_multi<<<gridDim, blockDim.x, 2 * blockDim.x * blockDim.y>>>(rows, columns, inners, blockDim.x, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled multi 2 matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
