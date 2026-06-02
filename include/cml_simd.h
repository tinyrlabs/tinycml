/**
 * @file cml_simd.h
 * @brief SIMD-accelerated hot-path operations for tinycml
 *
 * Provides NEON (aarch64), SSE2 (x86_64), and scalar fallback
 * implementations of common linear-algebra primitives used
 * throughout the library (dot products, distances, vector ops,
 * matrix-vector multiply).
 */

#ifndef CML_SIMD_H
#define CML_SIMD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Set to 1 when any SIMD path is compiled in, 0 for scalar fallback. */
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__SSE2__)
#define CML_USE_SIMD 1
#else
#define CML_USE_SIMD 0
#endif

/**
 * Compute the dot product of two vectors.
 * @param a First vector (length n)
 * @param b Second vector (length n)
 * @param n Number of elements
 * @return a · b
 */
double cml_simd_dot_product(const double *a, const double *b, size_t n);

/**
 * Compute the Euclidean (L2) distance between two vectors.
 * @param a First vector (length n)
 * @param b Second vector (length n)
 * @param n Number of elements
 * @return sqrt(sum((a_i - b_i)^2))
 */
double cml_simd_euclidean_distance(const double *a, const double *b, size_t n);

/**
 * Add two vectors element-wise: dst = a + b.
 * @param dst Output vector (length n, may alias a or b)
 * @param a First input vector
 * @param b Second input vector
 * @param n Number of elements
 */
void cml_simd_vec_add(double *dst, const double *a, const double *b, size_t n);

/**
 * Scale a vector by a scalar: dst = a * s.
 * @param dst Output vector (length n, may alias a)
 * @param a Input vector
 * @param s Scalar multiplier
 * @param n Number of elements
 */
void cml_simd_vec_scale(double *dst, const double *a, double s, size_t n);

/**
 * Matrix-vector multiplication: out = mat * vec.
 * mat is stored row-major with `cols` columns.
 * @param out   Output vector (length rows)
 * @param mat   Matrix (rows × cols, row-major)
 * @param vec   Input vector (length cols)
 * @param rows  Number of rows in mat
 * @param cols  Number of columns in mat
 */
void cml_simd_mat_vec_multiply(double *out, const double *mat,
                               const double *vec, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* CML_SIMD_H */
