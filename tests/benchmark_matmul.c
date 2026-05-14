/**
 * @file benchmark_matmul.c
 * @brief Benchmark for cache-optimized matrix multiplication
 */

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static void fill_random(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        /* Random double in [-1.0, 1.0] */
        m->data[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    }
}

static Matrix *create_identity(size_t n) {
    Matrix *m = matrix_alloc(n, n);
    if (!m) return NULL;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m->data[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return m;
}

static double benchmark_matmul(size_t n) {
    Matrix *a = matrix_alloc(n, n);
    Matrix *b = matrix_alloc(n, n);
    fill_random(a);
    fill_random(b);

    clock_t start = clock();
    Matrix *result = matrix_matmul(a, b);
    clock_t end = clock();

    double ms = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);

    return ms;
}

static int verify_identity(void) {
    size_t n = 50;
    Matrix *a = matrix_alloc(n, n);
    fill_random(a);
    Matrix *eye = create_identity(n);

    /* A * I should equal A */
    Matrix *result = matrix_matmul(a, eye);

    double max_err = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        double err = fabs(result->data[i] - a->data[i]);
        if (err > max_err) max_err = err;
    }

    matrix_free(a);
    matrix_free(eye);
    matrix_free(result);

    if (max_err > 1e-10) {
        fprintf(stderr, "  FAIL: max error = %e (expected < 1e-10)\n", max_err);
        return 1;
    }
    printf("  PASS: A * I == A (max error = %e)\n", max_err);
    return 0;
}

int main(void) {
    srand(42);

    printf("=== Matrix Multiplication Benchmark ===\n\n");

    /* Verify correctness */
    printf("Correctness check (identity multiply):\n");
    if (verify_identity() != 0) {
        return 1;
    }
    printf("\n");

    /* Benchmark different sizes */
    size_t sizes[] = {100, 200, 500};
    int num_sizes = 3;

    printf("Timing results (O2 optimized):\n");
    for (int i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        double ms = benchmark_matmul(n);
        printf("  %3zux%-3zu : %8.2f ms\n", n, n, ms);
    }

    printf("\nDone.\n");
    return 0;
}
