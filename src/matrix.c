/**
 * @file matrix.c
 * @brief Implementation of matrix operations
 */

#include "matrix.h"
#include "cml_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

Matrix* matrix_alloc(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }

    Matrix *m = malloc(sizeof(Matrix));
    if (!m) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_copy(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *copy = matrix_alloc(m->rows, m->cols);
    if (!copy) {
        return NULL;
    }

    memcpy(copy->data, m->data, m->rows * m->cols * sizeof(double));
    return copy;
}

double matrix_get(const Matrix *m, size_t i, size_t j) {
    if (!m || i >= m->rows || j >= m->cols) {
        cml_set_error(CML_ERROR_OUT_OF_BOUNDS,
                      "matrix_get: out of bounds or NULL (%zu, %zu) vs (%zu, %zu)",
                      i, j, m ? m->rows : 0, m ? m->cols : 0);
        return 0.0;  /* Safe default, avoids crash */
    }
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, size_t i, size_t j, double val) {
    if (!m || i >= m->rows || j >= m->cols) {
        cml_set_error(CML_ERROR_OUT_OF_BOUNDS,
                      "matrix_set: out of bounds or NULL (%zu, %zu) vs (%zu, %zu)",
                      i, j, m ? m->rows : 0, m ? m->cols : 0);
        return;  /* No-op on invalid access */
    }
    m->data[i * m->cols + j] = val;
}

Matrix* matrix_add(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_add: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix* matrix_sub(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_sub: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix* matrix_mul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_mul: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

Matrix* matrix_scale(const Matrix *m, double scalar) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->rows, m->cols);
    if (!result) {
        return NULL;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = m->data[i] * scalar;
    }

    return result;
}

#ifdef _OPENMP
static Matrix* matrix_matmul_omp(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->cols != b->rows) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_matmul: dimension mismatch (%zu x %zu) * (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    size_t M = a->rows, K = a->cols, N = b->cols;
    Matrix *result = matrix_alloc(M, N);
    if (!result) {
        return NULL;
    }

    /* Allocate transposed B for cache-friendly column access */
    double *b_colmajor = malloc(K * N * sizeof(double));
    if (!b_colmajor) {
        matrix_free(result);
        return NULL;
    }

    /* Transpose B: b_colmajor[j * K + i] = b->data[i * N + j] */
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b_colmajor[j * K + i] = b->data[i * N + j];
        }
    }

    /* i-k-j loop with OpenMP on outer loop */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double a_ik = a->data[i * K + k];
            for (size_t j = 0; j < N; j++) {
                result->data[i * N + j] += a_ik * b_colmajor[j * K + k];
            }
        }
    }

    free(b_colmajor);
    return result;
}
#endif /* _OPENMP */

Matrix* matrix_matmul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        cml_set_error(CML_ERROR_NULL_PTR, "matrix_matmul: NULL input");
        return NULL;
    }

    if (a->cols != b->rows) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_matmul: dimension mismatch (%zu x %zu) * (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

#ifdef _OPENMP
    if (omp_get_max_threads() > 1) return matrix_matmul_omp(a, b);
#endif

    size_t M = a->rows, K = a->cols, N = b->cols;
    Matrix *result = matrix_alloc(M, N);
    if (!result) {
        return NULL;
    }

    /* Allocate transposed B for cache-friendly column access */
    double *b_colmajor = malloc(K * N * sizeof(double));
    if (!b_colmajor) {
        matrix_free(result);
        return NULL;
    }

    /* Transpose B: b_colmajor[j * K + i] = b->data[i * N + j] */
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b_colmajor[j * K + i] = b->data[i * N + j];
        }
    }

    /* i-k-j loop: access A row-wise and B^T row-wise (both sequential) */
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double a_ik = a->data[i * K + k];
            for (size_t j = 0; j < N; j++) {
                result->data[i * N + j] += a_ik * b_colmajor[j * K + k];
            }
        }
    }

    free(b_colmajor);
    return result;
}

Matrix* matrix_transpose(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->cols, m->rows);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }

    return result;
}

void matrix_print(const Matrix *m) {
    if (!m) {
        printf("(null matrix)\n");
        return;
    }

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        printf("  [");
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void matrix_fill(Matrix *m, double val) {
    if (!m) {
        return;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] = val;
    }
}

Matrix* matrix_identity(size_t n) {
    if (n == 0) {
        return NULL;
    }

    Matrix *m = matrix_alloc(n, n);
    if (!m) {
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        m->data[i * n + i] = 1.0;
    }

    return m;
}
