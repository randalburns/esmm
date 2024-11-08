#include <iostream>
#include "esmm_cpu.h"

void testSquare()
{
    // Define 4x4 matrices A and B, and an output matrix C

    constexpr int rows = 4;
    constexpr int columns = 4;
    constexpr int inners = 4;

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 2.0, 3.0, 4.0,
                              2.0, 3.0, 4.0, 5.0,
                              3.0, 4.0, 5.0, 6.0,
                              4.0, 5.0, 6.0, 7.0};

    float B[inners * columns] = {1.0, 2.0, 3.0, 4.0,
                                  2.0, 3.0, 4.0, 5.0,
                                  3.0, 4.0, 5.0, 6.0,
                                  4.0, 5.0, 6.0, 7.0};

    float C[rows * columns];
    zeroMatrix<rows,columns>(C);
    // Call the baseMM function to multiply A and B, storing the result in C
    baseMM<rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 2, 2, 2>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 4, 4, 4>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 4, 2, 4>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 4, 2, 2>(A, B, C);
    printMatrix<rows, columns>(C);
}

void testRect()
{
    // Define 4x4 matrices with 2 inners

    constexpr int rows = 4;
    constexpr int columns = 4;
    constexpr int inners = 2;

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 2.0,
                              2.0, 3.0,
                              3.0, 4.0,
                              4.0, 5.0};

    float B[inners * columns] = {1.0, 2.0, 3.0, 4.0,
                                 2.0, 3.0, 4.0, 5.0 };

    float C[rows * columns];
    zeroMatrix<rows,columns>(C);
    // Call the baseMM function to multiply A and B, storing the result in C
    baseMM<rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 2, 2, 2>(A, B, C);
    printMatrix<rows, columns>(C);
}

void testRect2()
{
    // Define 6x6 output matrix with 3 inners

    constexpr int rows = 6;
    constexpr int columns = 6;
    constexpr int inners = 3;

     // Initialize matrices A and B with some values
    float A[rows * inners] = {1.0, 2.0, 3.0,
                              2.0, 3.0, 4.0,
                              3.0, 4.0, 5.0,
                              4.0, 5.0, 6.0,
                              5.0, 6.0, 7.0,
                              6.0, 7.0, 8.0};

    float B[inners * columns] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    float C[rows * columns];
    zeroMatrix<rows,columns>(C);
    // Call the baseMM function to multiply A and B, storing the result in C
    baseMM<rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 3, 3, 3>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    tiledMM<rows, columns, inners, 2, 2, 3>(A, B, C);
    printMatrix<rows, columns>(C);
}

int main()
{
    testSquare();
 //   testRect();
 //   testRect2();

    return 0;
}
