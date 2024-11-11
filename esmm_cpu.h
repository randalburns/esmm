#ifndef ESMMCPU_H
#define ESMMCPU_H

#include <iostream>
#include <cassert>

/**
 * @brief Simple matrix multiplication of two matrices A and B and stores the result in matrix C.
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

/**
 * @brief Tiled matrix multiplication of two matrices A and B and stores the result in matrix C.
 *
 * This function performs matrix multiplication of two matrices A and B using a triple-nested loop.
 * Matrix A has dimensions [rows x inners], matrix B has dimensions [inners x columns], and the result
 * matrix C has dimensions [rows x columns]. The result is stored in C without clearing it beforehand,
 * so C should be initialized before calling this function.
 *
 * @tparam rows Number of rows in matrix A and matrix C.
 * @tparam columns Number of columns in matrix B and matrix C.
 * @tparam inners Number of columns in matrix A and rows in matrix B.
 * @tparam rTileSize tile size in rows dimension
 * @tparam cTileSize tile size in columns dimension
 * @tparam iTileSize tile size in inners dimension
 * @param[in] A Pointer to the first matrix (of size rows * inners).
 * @param[in] B Pointer to the second matrix (of size inners * columns).
 * @param[out] C Pointer to the result matrix (of size rows * columns), where the computed result will be stored.
 */
template <int rows, int columns, int inners, int rTileSize, int cTileSize, int iTileSize>
inline void tiledMM(const float *A, const float *B, float *C)
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

/**
 * @brief Unrolled and tiled matrix multiplication of two matrices A and B and stores the result in matrix C.
 *
 * This function unrolls the columns 4 times. It assumes a cTileSize of 4.
 * This is a precursor implementation for CUDA in which the threads will be per innner.
 * 
 * This function performs matrix multiplication of two matrices A and B using a triple-nested loop.
 * Matrix A has dimensions [rows x inners], matrix B has dimensions [inners x columns], and the result
 * matrix C has dimensions [rows x columns]. The result is stored in C without clearing it beforehand,
 * so C should be initialized before calling this function.
 *
 * @tparam rows Number of rows in matrix A and matrix C.
 * @tparam columns Number of columns in matrix B and matrix C.
 * @tparam inners Number of columns in matrix A and rows in matrix B.
 * @tparam rTileSize tile size in rows dimension
 * @tparam cTileSize tile size in columns dimension
 * @tparam iTileSize tile size in inners dimension
 * @param[in] A Pointer to the first matrix (of size rows * inners).
 * @param[in] B Pointer to the second matrix (of size inners * columns).
 * @param[out] C Pointer to the result matrix (of size rows * columns), where the computed result will be stored.
 */
template <int rows, int columns, int inners, int rTileSize, int cTileSize, int iTileSize>
inline void tiledMM_unroll4(const float *A, const float *B, float *C)
{
    assert(cTileSize==4);
    for (int rTileOff = 0; rTileOff < rows; rTileOff += rTileSize) {
    for (int iTileOff = 0; iTileOff < inners; iTileOff += iTileSize) {
    for (int cTileOff = 0; cTileOff < columns; cTileOff += cTileSize)
    {
        for (int rowoff = 0; rowoff < rTileSize; rowoff++)
        {
            for (int inneroff = 0; inneroff < iTileSize; inneroff++)
            {
                int row = rTileOff + rowoff;
                int inner = iTileOff + inneroff;

                // Unrolled coloff loop with columns == 4
                C[row * columns + (cTileOff + 0)] += A[row * inners + inner] * B[inner * columns + (cTileOff + 0)];
                C[row * columns + (cTileOff + 1)] += A[row * inners + inner] * B[inner * columns + (cTileOff + 1)];
                C[row * columns + (cTileOff + 2)] += A[row * inners + inner] * B[inner * columns + (cTileOff + 2)];
                C[row * columns + (cTileOff + 3)] += A[row * inners + inner] * B[inner * columns + (cTileOff + 3)];
            }
        }
    }}}
}

template <int rows, int columns>
inline void printMatrix (float *mat)
{
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
inline void zeroMatrix (float *mat) {
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            mat[row * columns + col] = 0;
        }
    }
}

#endif // ESMMCPU_H
