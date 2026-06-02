#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/**
 * @file test_simd.c
 * @brief Tests for SIMD-accelerated operations
 *
 * Validates correctness of all cml_simd functions and reports
 * whether the SIMD or scalar path was compiled in.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "cml_simd.h"

/* ---- helpers ---------------------------------------------------- */

static int approx_eq(double a, double b, double tol)
{
    return fabs(a - b) < tol;
}

static int check(const char *label, int condition)
{
    if (condition)
        printf("  PASS  %s\n", label);
    else
        printf("  FAIL  %s\n", label);
    return !condition;
}

static double scalar_dot(const double *a, const double *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double scalar_dist(const double *a, const double *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}

static void scalar_add(double *dst, const double *a, const double *b, size_t n)
{
    for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

static void scalar_scale(double *dst, const double *a, double s, size_t n)
{
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * s;
}

static void scalar_mat_vec(double *out, const double *mat,
                           const double *vec, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        double s = 0.0;
        for (size_t c = 0; c < cols; c++)
            s += mat[r * cols + c] * vec[c];
        out[r] = s;
    }
}

/* ---- random vector helpers -------------------------------------- */

static double *alloc_vec(size_t n)
{
    double *v = (double *)malloc(n * sizeof(double));
    if (!v) { fprintf(stderr, "OOM\n"); exit(1); }
    return v;
}

static void fill_random(double *v, size_t n, unsigned *seed)
{
    for (size_t i = 0; i < n; i++)
        v[i] = ((double)rand_r(seed) / RAND_MAX) * 2.0 - 1.0;
}

/* ---- micro-benchmark helper ------------------------------------- */

static double elapsed_sec(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) / 1e9;
}

/* ---- tests ------------------------------------------------------ */

static int test_dot_product(size_t n, const char *label)
{
    int fails = 0;
    unsigned seed = (unsigned)(uintptr_t)label ^ 42;
    double *a = alloc_vec(n);
    double *b = alloc_vec(n);
    fill_random(a, n, &seed);
    fill_random(b, n, &seed);

    double expected = scalar_dot(a, b, n);
    double got = cml_simd_dot_product(a, b, n);
    fails += check(label, approx_eq(got, expected, 1e-9));

    free(a);
    free(b);
    return fails;
}

static int test_euclidean_distance(size_t n, const char *label)
{
    int fails = 0;
    unsigned seed = (unsigned)(uintptr_t)label ^ 77;
    double *a = alloc_vec(n);
    double *b = alloc_vec(n);
    fill_random(a, n, &seed);
    fill_random(b, n, &seed);

    double expected = scalar_dist(a, b, n);
    double got = cml_simd_euclidean_distance(a, b, n);
    fails += check(label, approx_eq(got, expected, 1e-9));

    free(a);
    free(b);
    return fails;
}

static int test_vec_add(size_t n, const char *label)
{
    int fails = 0;
    unsigned seed = (unsigned)(uintptr_t)label ^ 13;
    double *a = alloc_vec(n);
    double *b = alloc_vec(n);
    double *dst = alloc_vec(n);
    double *ref = alloc_vec(n);
    fill_random(a, n, &seed);
    fill_random(b, n, &seed);

    scalar_add(ref, a, b, n);
    cml_simd_vec_add(dst, a, b, n);

    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (!approx_eq(dst[i], ref[i], 1e-12)) { ok = 0; break; }
    }
    fails += check(label, ok);

    free(a); free(b); free(dst); free(ref);
    return fails;
}

static int test_vec_scale(size_t n, const char *label)
{
    int fails = 0;
    unsigned seed = (unsigned)(uintptr_t)label ^ 55;
    double *a = alloc_vec(n);
    double *dst = alloc_vec(n);
    double *ref = alloc_vec(n);
    fill_random(a, n, &seed);

    double s = 3.14;
    scalar_scale(ref, a, s, n);
    cml_simd_vec_scale(dst, a, s, n);

    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (!approx_eq(dst[i], ref[i], 1e-12)) { ok = 0; break; }
    }
    fails += check(label, ok);

    free(a); free(dst); free(ref);
    return fails;
}

static int test_mat_vec_multiply(size_t rows, size_t cols, const char *label)
{
    int fails = 0;
    unsigned seed = (unsigned)(uintptr_t)label ^ 99;
    double *mat = alloc_vec(rows * cols);
    double *vec = alloc_vec(cols);
    double *out = alloc_vec(rows);
    double *ref = alloc_vec(rows);
    fill_random(mat, rows * cols, &seed);
    fill_random(vec, cols, &seed);

    scalar_mat_vec(ref, mat, vec, rows, cols);
    cml_simd_mat_vec_multiply(out, mat, vec, rows, cols);

    int ok = 1;
    for (size_t i = 0; i < rows; i++) {
        if (!approx_eq(out[i], ref[i], 1e-9)) { ok = 0; break; }
    }
    fails += check(label, ok);

    free(mat); free(vec); free(out); free(ref);
    return fails;
}

/* ---- benchmark -------------------------------------------------- */

static void bench_dot_product(void)
{
    const size_t N = 1024 * 1024;  /* 1 M elements */
    const int ITERS = 200;
    unsigned seed = 12345;
    double *a = alloc_vec(N);
    double *b = alloc_vec(N);
    fill_random(a, N, &seed);
    fill_random(b, N, &seed);

    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++)
        scalar_dot(a, b, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_scalar = elapsed_sec(t0, t1);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++)
        cml_simd_dot_product(a, b, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_simd = elapsed_sec(t0, t1);

    printf("\n  dot product benchmark  (N=%zu, iters=%d)\n", N, ITERS);
    printf("    scalar : %.4f s\n", t_scalar);
    printf("    simd   : %.4f s\n", t_simd);
    if (t_simd > 0)
        printf("    speedup: %.2fx\n", t_scalar / t_simd);

    free(a); free(b);
}

/* ---- main ------------------------------------------------------- */

int main(void)
{
    int fails = 0;

    printf("=== SIMD tests ===\n");
    printf("CML_USE_SIMD = %d  (0=scalar, 1=SIMD path active)\n\n", CML_USE_SIMD);

    /* --- small / odd sizes for tail handling --- */
    fails += test_dot_product(1,    "dot  n=1   (odd)");
    fails += test_dot_product(3,    "dot  n=3   (odd)");
    fails += test_dot_product(7,    "dot  n=7   (odd)");
    fails += test_dot_product(15,   "dot  n=15  (odd)");

    fails += test_euclidean_distance(1,    "dist n=1   (odd)");
    fails += test_euclidean_distance(5,    "dist n=5   (odd)");
    fails += test_euclidean_distance(9,    "dist n=9   (odd)");
    fails += test_euclidean_distance(13,   "dist n=13  (odd)");

    fails += test_vec_add(1,  "vadd n=1   (odd)");
    fails += test_vec_add(3,  "vadd n=3   (odd)");
    fails += test_vec_add(11, "vadd n=11  (odd)");

    fails += test_vec_scale(1,  "vscl n=1   (odd)");
    fails += test_vec_scale(3,  "vscl n=3   (odd)");
    fails += test_vec_scale(11, "vscl n=11  (odd)");

    fails += test_mat_vec_multiply(3, 5,  "matvec 3x5  (odd)");
    fails += test_mat_vec_multiply(7, 3,  "matvec 7x3  (odd)");

    /* --- even sizes --- */
    fails += test_dot_product(2,    "dot  n=2   (even)");
    fails += test_dot_product(8,    "dot  n=8   (even)");
    fails += test_dot_product(16,   "dot  n=16  (even)");

    fails += test_euclidean_distance(2,    "dist n=2   (even)");
    fails += test_euclidean_distance(8,    "dist n=8   (even)");
    fails += test_euclidean_distance(16,   "dist n=16  (even)");

    fails += test_vec_add(2,  "vadd n=2   (even)");
    fails += test_vec_add(8,  "vadd n=8   (even)");

    fails += test_vec_scale(2,  "vscl n=2   (even)");
    fails += test_vec_scale(8,  "vscl n=8   (even)");

    fails += test_mat_vec_multiply(4, 4,  "matvec 4x4  (even)");
    fails += test_mat_vec_multiply(8, 8,  "matvec 8x8  (even)");

    /* --- large vectors --- */
    fails += test_dot_product(512,      "dot  n=512");
    fails += test_dot_product(1023,     "dot  n=1023 (odd)");
    fails += test_dot_product(2048,     "dot  n=2048");

    fails += test_euclidean_distance(512,      "dist n=512");
    fails += test_euclidean_distance(1023,     "dist n=1023 (odd)");
    fails += test_euclidean_distance(2048,     "dist n=2048");

    fails += test_vec_add(512,  "vadd n=512");
    fails += test_vec_add(1023, "vadd n=1023 (odd)");

    fails += test_vec_scale(512,  "vscl n=512");
    fails += test_vec_scale(1023, "vscl n=1023 (odd)");

    fails += test_mat_vec_multiply(64, 512,  "matvec 64x512");
    fails += test_mat_vec_multiply(100, 511, "matvec 100x511 (odd cols)");

    /* --- benchmark --- */
    bench_dot_product();

    printf("\n");
    if (fails)
        printf("FAILED %d tests\n", fails);
    else
        printf("All SIMD tests passed!\n");

    return fails ? 1 : 0;
}
