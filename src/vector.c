/**
 * @file vector.c
 * @brief Implementation of vector operations
 */

#include "vector.h"
#include "cml_error.h"
#include <math.h>
#include <stdio.h>

double vector_dot(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        cml_set_error(CML_ERROR_NULL_PTR, "vector_dot: NULL input");
        return 0.0;
    }

    size_t n_a = a->rows * a->cols;
    size_t n_b = b->rows * b->cols;

    if (n_a != n_b) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH, "vector_dot: size mismatch (%zu vs %zu)", n_a, n_b);
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n_a; i++) {
        sum += a->data[i] * b->data[i];
    }

    return sum;
}

double vector_norm(const Matrix *v) {
    if (!v) {
        return 0.0;
    }

    double sum = 0.0;
    size_t n = v->rows * v->cols;

    for (size_t i = 0; i < n; i++) {
        sum += v->data[i] * v->data[i];
    }

    return sqrt(sum);
}

Matrix* vector_scale(const Matrix *v, double scalar) {
    return matrix_scale(v, scalar);
}

Matrix* vector_add(const Matrix *a, const Matrix *b) {
    return matrix_add(a, b);
}

Matrix* vector_sub(const Matrix *a, const Matrix *b) {
    return matrix_sub(a, b);
}
