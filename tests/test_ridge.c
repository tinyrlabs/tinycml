/**
 * @file test_ridge.c
 * @brief Unit tests for Ridge regression
 */

#include "test_harness.h"
#include "matrix.h"
#include "ridge.h"
#include "estimator.h"
#include <math.h>

TEST(test_ridge_create_free) {
    RidgeModel *m = ridge_model_create();
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->base.is_fitted, 0);
    ASSERT_EQ(m->base.task, TASK_REGRESSION);
    ASSERT_NEAR(m->alpha, 1.0, 1e-12);
    ASSERT_NULL(m->weights);

    ridge_free((Estimator *)m);
}

TEST(test_ridge_fit_predict) {
    /* Linear data with noise: y ~ 3*x1 + 2*x2 + 5 + small noise */
    int n = 50;
    int p = 2;
    Matrix *X = matrix_alloc(n, p);
    Matrix *y = matrix_alloc(n, 1);

    /* Deterministic "noise" via simple pattern */
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        double x1 = (double)(i - n / 2) / 10.0;
        double x2 = (double)(i % 7) - 3.0;
        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);

        /* Pseudo-random noise in [-0.5, 0.5] */
        seed = seed * 1103515245 + 12345;
        double noise = ((double)((seed >> 16) & 0x7fff) / 16384.0) - 0.5;

        double val = 3.0 * x1 + 2.0 * x2 + 5.0 + noise * 0.3;
        matrix_set(y, i, 0, val);
    }

    RidgeModel *m = ridge_model_create();
    m->alpha = 0.1;  /* small regularization */
    Estimator *e = ridge_fit((Estimator *)m, X, y);
    ASSERT_NOT_NULL(e);
    ASSERT_EQ(m->base.is_fitted, 1);

    /* Check weights are close to [3, 2] */
    ASSERT_NEAR(m->weights->data[0], 3.0, 0.5);
    ASSERT_NEAR(m->weights->data[1], 2.0, 0.5);

    /* R² > 0.8 */
    double r2 = ridge_score((Estimator *)m, X, y);
    ASSERT(r2 > 0.8);

    matrix_free(X);
    matrix_free(y);
    ridge_free((Estimator *)m);
}

TEST(test_ridge_regularization) {
    /* High alpha should produce smaller weights than OLS (approximated by low alpha) */
    int n = 20;
    int p = 3;
    Matrix *X = matrix_alloc(n, p);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n; i++) {
        matrix_set(X, i, 0, (double)i / 5.0);
        matrix_set(X, i, 1, (double)(i * i) / 50.0);
        matrix_set(X, i, 2, sin((double)i));
        double val = 2.0 * matrix_get(X, i, 0)
                   + 3.0 * matrix_get(X, i, 1)
                   - 1.0 * matrix_get(X, i, 2);
        matrix_set(y, i, 0, val);
    }

    /* Low alpha (near-OLS) */
    RidgeModel *m_low = ridge_model_create();
    m_low->alpha = 1e-8;
    ridge_fit((Estimator *)m_low, X, y);
    ASSERT_NOT_NULL(m_low->weights);

    /* High alpha */
    RidgeModel *m_high = ridge_model_create();
    m_high->alpha = 1000.0;
    ridge_fit((Estimator *)m_high, X, y);
    ASSERT_NOT_NULL(m_high->weights);

    /* Compute L2 norm of weights */
    double norm_low  = 0.0, norm_high = 0.0;
    for (int j = 0; j < p; j++) {
        norm_low  += m_low->weights->data[j]  * m_low->weights->data[j];
        norm_high += m_high->weights->data[j] * m_high->weights->data[j];
    }
    norm_low  = sqrt(norm_low);
    norm_high = sqrt(norm_high);

    /* High alpha => smaller weights */
    ASSERT(norm_high < norm_low);

    matrix_free(X);
    matrix_free(y);
    ridge_free((Estimator *)m_low);
    ridge_free((Estimator *)m_high);
}

int main(void) {
    printf("Ridge Regression Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_ridge_create_free);
    RUN_TEST(test_ridge_fit_predict);
    RUN_TEST(test_ridge_regularization);

    TEST_SUMMARY();
}
