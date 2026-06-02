/**
 * @file cml_simd.c
 * @brief SIMD-accelerated hot-path operations — implementation
 *
 * Compile-time dispatch:
 *   - aarch64 → ARM NEON float64 intrinsics (full fp64 on aarch64)
 *   - x86_64  → SSE2 double intrinsics
 *   - other   → scalar C loops
 */

#include "cml_simd.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/*  ARM NEON path  (aarch64 has full float64 NEON)                    */
/* ------------------------------------------------------------------ */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

/* SIMD width for float64 on aarch64: 128 bits → 2 × f64 */
#define CML_SIMD_WIDTH 2

double cml_simd_dot_product(const double *a, const double *b, size_t n)
{
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        acc = vfmaq_f64(acc, va, vb);
    }
    /* horizontal sum */
    double tmp[2];
    vst1q_f64(tmp, acc);
    double sum = tmp[0] + tmp[1];
    /* tail */
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

double cml_simd_euclidean_distance(const double *a, const double *b, size_t n)
{
    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t diff = vsubq_f64(va, vb);
        acc = vfmaq_f64(acc, diff, diff);
    }
    double tmp[2];
    vst1q_f64(tmp, acc);
    double sum = tmp[0] + tmp[1];
    for (; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

void cml_simd_vec_add(double *dst, const double *a, const double *b, size_t n)
{
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(dst + i, vaddq_f64(va, vb));
    }
    for (; i < n; i++)
        dst[i] = a[i] + b[i];
}

void cml_simd_vec_scale(double *dst, const double *a, double s, size_t n)
{
    float64x2_t vs = vdupq_n_f64(s);
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(dst + i, vmulq_f64(va, vs));
    }
    for (; i < n; i++)
        dst[i] = a[i] * s;
}

void cml_simd_mat_vec_multiply(double *out, const double *mat,
                               const double *vec, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        const double *row = mat + r * cols;
        out[r] = cml_simd_dot_product(row, vec, cols);
    }
}

/* ------------------------------------------------------------------ */
/*  SSE2 path                                                         */
/* ------------------------------------------------------------------ */
#elif defined(__SSE2__)

#include <emmintrin.h>

#define CML_SIMD_WIDTH 2

double cml_simd_dot_product(const double *a, const double *b, size_t n)
{
    __m128d acc = _mm_setzero_pd();
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
    }
    /* horizontal sum */
    __m128d shuffle = _mm_shuffle_pd(acc, acc, 0x01);
    __m128d sum_vec = _mm_add_pd(acc, shuffle);
    double sum;
    _mm_store_sd(&sum, sum_vec);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

double cml_simd_euclidean_distance(const double *a, const double *b, size_t n)
{
    __m128d acc = _mm_setzero_pd();
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d diff = _mm_sub_pd(va, vb);
        acc = _mm_add_pd(acc, _mm_mul_pd(diff, diff));
    }
    __m128d shuffle = _mm_shuffle_pd(acc, acc, 0x01);
    __m128d sum_vec = _mm_add_pd(acc, shuffle);
    double sum;
    _mm_store_sd(&sum, sum_vec);
    for (; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

void cml_simd_vec_add(double *dst, const double *a, const double *b, size_t n)
{
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(va, vb));
    }
    for (; i < n; i++)
        dst[i] = a[i] + b[i];
}

void cml_simd_vec_scale(double *dst, const double *a, double s, size_t n)
{
    __m128d vs = _mm_set1_pd(s);
    size_t i = 0;
    for (; i + CML_SIMD_WIDTH <= n; i += CML_SIMD_WIDTH) {
        __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(dst + i, _mm_mul_pd(va, vs));
    }
    for (; i < n; i++)
        dst[i] = a[i] * s;
}

void cml_simd_mat_vec_multiply(double *out, const double *mat,
                               const double *vec, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        const double *row = mat + r * cols;
        out[r] = cml_simd_dot_product(row, vec, cols);
    }
}

/* ------------------------------------------------------------------ */
/*  Scalar fallback                                                   */
/* ------------------------------------------------------------------ */
#else  /* no SIMD */

double cml_simd_dot_product(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

double cml_simd_euclidean_distance(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

void cml_simd_vec_add(double *dst, const double *a, const double *b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
}

void cml_simd_vec_scale(double *dst, const double *a, double s, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] * s;
}

void cml_simd_mat_vec_multiply(double *out, const double *mat,
                               const double *vec, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        const double *row = mat + r * cols;
        double sum = 0.0;
        for (size_t c = 0; c < cols; c++)
            sum += row[c] * vec[c];
        out[r] = sum;
    }
}

#endif /* SIMD dispatch */
