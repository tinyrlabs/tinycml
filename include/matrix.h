/**
 * @file matrix.h
 * @brief Matrix data structure and operations for ML library
 *
 * Provides core matrix functionality including allocation, arithmetic,
 * and transformations. Uses row-major storage order.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/**
 * @brief Matrix structure with row-major storage
 */
typedef struct {
    size_t rows;    /**< Number of rows */
    size_t cols;    /**< Number of columns */
    double *data;   /**< Row-major data array */
} Matrix;

/* Memory Management */

/**
 * @brief Allocate a zero-initialized matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to allocated matrix, or NULL on failure
 */
Matrix* matrix_alloc(size_t rows, size_t cols);

/**
 * @brief Free matrix memory
 * @param m Matrix to free
 */
void matrix_free(Matrix *m);

/**
 * @brief Create a deep copy of a matrix
 * @param m Matrix to copy
 * @return Pointer to new matrix, or NULL on failure
 */
Matrix* matrix_copy(const Matrix *m);

/* Element Access */

/**
 * @brief Get element at position (i, j)
 * @param m Matrix
 * @param i Row index
 * @param j Column index
 * @return Element value
 */
double matrix_get(const Matrix *m, size_t i, size_t j);

/**
 * @brief Set element at position (i, j)
 * @param m Matrix
 * @param i Row index
 * @param j Column index
 * @param val Value to set
 */
void matrix_set(Matrix *m, size_t i, size_t j, double val);

/* Arithmetic Operations */

/**
 * @brief Element-wise addition: C = A + B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_add(const Matrix *a, const Matrix *b);

/**
 * @brief Element-wise subtraction: C = A - B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_sub(const Matrix *a, const Matrix *b);

/**
 * @brief Element-wise multiplication (Hadamard product): C = A .* B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_mul(const Matrix *a, const Matrix *b);

/**
 * @brief Scalar multiplication: C = scalar * A
 * @param m Matrix
 * @param scalar Scalar value
 * @return Result matrix, or NULL on failure
 */
Matrix* matrix_scale(const Matrix *m, double scalar);

/**
 * @brief Matrix multiplication: C = A * B
 * @param a First matrix (m x k)
 * @param b Second matrix (k x n)
 * @return Result matrix (m x n), or NULL on dimension mismatch
 */
Matrix* matrix_matmul(const Matrix *a, const Matrix *b);

/* Transformations */

/**
 * @brief Transpose matrix
 * @param m Matrix to transpose
 * @return Transposed matrix, or NULL on failure
 */
Matrix* matrix_transpose(const Matrix *m);

/* Utilities */

/**
 * @brief Print matrix to stdout
 * @param m Matrix to print
 */
void matrix_print(const Matrix *m);

/**
 * @brief Fill matrix with a constant value
 * @param m Matrix to fill
 * @param val Value to fill with
 */
void matrix_fill(Matrix *m, double val);

/**
 * @brief Create an identity matrix
 * @param n Size of the square matrix
 * @return n x n identity matrix, or NULL on failure
 */
Matrix* matrix_identity(size_t n);

#endif /* MATRIX_H */
