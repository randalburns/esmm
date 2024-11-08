
#include <iostream>

/**
 * @brief Performs matrix multiplication of two matrices A and B and stores the result in matrix C.
 *
 * This function performs matrix multiplication of two matrices A and B using a triple-nested loop.
 * Matrix A has dimensions [rows x inners], matrix B has dimensions [inners x columns], and the result
 * matrix C has dimensions [rows x columns]. The result is stored in C without clearing it beforehand, 
 * so C should be initialized before calling this function.
 *
 * @tparam rows Number of rows in matrix A and matrix C.
 * @tparam columns Number of columns in matrix B and matrix C.
 * @tparam inners Number of columns in matrix A and rows in matrix B.
 * @param[in] A Pointer to the first matrix (of size rows * inners).
 * @param[in] B Pointer to the second matrix (of size inners * columns).
 * @param[out] C Pointer to the result matrix (of size rows * columns), where the computed result will be stored.
 */

template <int rows, int columns, int inners>
inline void baseMM(const float *A, const float *B, float *C)
{
    for (int row = 0; row < rows; row++)
    {
        for (int inner = 0; inner < inners; inner++)
        {
            for (int col = 0; col < columns; col++)
            {
                C[row * columns + col] +=
                    A[row * inners + inner] * B[inner * columns + col];
            }
        }
    }
}

template <int rows, int columns, int inners, int rTileSize, int cTileSize, int iTileSize>
inline void MMBTiled(const float *A, const float *B, float *C)
{
    for (int rTileOff = 0; rTileOff < rows; rTileOff += rTileSize) {
    for (int iTileOff = 0; iTileOff < inners; iTileOff += iTileSize) {
    for (int cTileOff = 0; cTileOff < columns; cTileOff += cTileSize)
    {
        for (int rowoff = 0; rowoff < rTileSize; rowoff++)
        {
            for (int inneroff = 0; inneroff < iTileSize; inneroff++)
            {
                for (int coloff = 0; coloff < cTileSize; coloff++)
                {
                    int row = rTileOff + rowoff;
                    int col = cTileOff + coloff;
                    int inner = iTileOff + inneroff;
                    C[row * columns + col] +=
                        A[row * inners + inner] *
                        B[inner * columns + col];
                }
            }
        }
    }}}
}

template <int rows, int columns>
inline void printMatrix (float *mat)
{
    // Print the result matrix C
    std::cout << "Result matrix C:" << std::endl;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            std::cout << mat[row * columns + col] << " ";
        }
        std::cout << std::endl;
    }
}

template <int rows, int columns>
inline void zeroMatrix (float *mat)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            mat[row * columns + col] = 0;
        }
    }
}

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
    MMBTiled<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 2, 2, 2>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 4, 4, 4>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 4, 2, 4>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 4, 2, 2>(A, B, C);
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
    MMBTiled<rows, columns, inners, rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 2, 2, 2>(A, B, C);
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
    MMBTiled<rows, columns, inners, rows, columns, inners>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 1, 1, 1>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 3, 3, 3>(A, B, C);
    printMatrix<rows, columns>(C);

    zeroMatrix<rows,columns>(C);
    MMBTiled<rows, columns, inners, 2, 2, 3>(A, B, C);
    printMatrix<rows, columns>(C);
}

int main()
{
    // Define 2x2 matrices A and B, and an output matrix C
    constexpr int rows = 4;
    constexpr int columns = 4;
    constexpr int inners = 4;

    testSquare();
    testRect();
    testRect2();

    return 0;
}