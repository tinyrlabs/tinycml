/**
 * @file vector.h
 * @brief Vector operations for ML library
 *
 * Provides common vector operations using Matrix (n x 1) representation.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include "matrix.h"

/**
 * @brief Compute dot product of two vectors
 * @param a First vector (n x 1 matrix)
 * @param b Second vector (n x 1 matrix)
 * @return Dot product value
 */
double vector_dot(const Matrix *a, const Matrix *b);

/**
 * @brief Compute L2 norm (Euclidean length) of a vector
 * @param v Vector (n x 1 matrix)
 * @return L2 norm
 */
double vector_norm(const Matrix *v);

/**
 * @brief Scale a vector by a scalar
 * @param v Vector (n x 1 matrix)
 * @param scalar Scalar value
 * @return Scaled vector, or NULL on failure
 */
Matrix* vector_scale(const Matrix *v, double scalar);

/**
 * @brief Add two vectors
 * @param a First vector
 * @param b Second vector
 * @return Sum vector, or NULL on failure
 */
Matrix* vector_add(const Matrix *a, const Matrix *b);

/**
 * @brief Subtract two vectors
 * @param a First vector
 * @param b Second vector
 * @return Difference vector, or NULL on failure
 */
Matrix* vector_sub(const Matrix *a, const Matrix *b);

#endif /* VECTOR_H */
