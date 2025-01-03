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


// Only works for up to 32 in oneMatrix

int main() {

    // Define 4x4 matrices A and B, and an output matrix C
    constexpr int rows = 32;
    constexpr int columns = 32;
    constexpr int inners = 32; 
     
    // base grid
    dim3 oneGrid(1,1);
    dim3 fullBlock(rows,columns);

    dim3 gridDim(8,8);
    dim3 blockDim(4,4);

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

    // Whole matrix in one kernel -- this is reference Cref
    cudaMemset(d_C, 0, Csize);
    esmm_naive<<<oneGrid, fullBlock>>>(rows, columns, inners, d_A, d_B, d_C);
    cudaMemcpy(Cref, d_C, Csize, cudaMemcpyDeviceToHost);
    cudaMemset(d_C, 0, Csize);

    // Check against CPU -- need a higher tolerance than 1e-6
    std::cout << "Base matches CPU = " << checkEqual ( rows, columns, Ccpu, Cref, 0.001 ) << std::endl;

    //printMatrix<rows, columns>(Cref);

    // Tiled naive
    cudaMemset(d_C, 0, Csize);
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled naive kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // sequential
    cudaMemset(d_C, 0, Csize);
    esmm_naive<<<oneGrid, fullBlock>>>(rows, columns, inners, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "One sequential kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // tiled sequential
    cudaMemset(d_C, 0, Csize);
    esmm_naive<<<gridDim, blockDim>>>(rows, columns, inners, d_A, d_B, d_C);
    cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost);
    std::cout << "Tiled sequential kernel matches = " << checkEqual ( rows, columns, Cref, C ) << std::endl;
    cudaMemset(d_C, 0, Csize);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
