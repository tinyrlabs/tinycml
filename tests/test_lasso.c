/**
 * @file test_lasso.c
 * @brief Unit tests for Lasso regression
 */

#include "test_harness.h"
#include "matrix.h"
#include "lasso.h"
#include "estimator.h"
#include <math.h>

TEST(test_lasso_create_free) {
    LassoModel *m = lasso_model_create();
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->base.is_fitted, 0);
    ASSERT_EQ(m->base.task, TASK_REGRESSION);
    ASSERT_NEAR(m->alpha, 1.0, 1e-12);
    ASSERT_NULL(m->weights);

    lasso_free((Estimator *)m);
}

TEST(test_lasso_fit_predict) {
    /* Sparse linear data: y = 5*x1 + 0*x2 + 0*x3 + 0*x4 + noise */
    int n = 60;
    int p = 4;
    Matrix *X = matrix_alloc(n, p);
    Matrix *y = matrix_alloc(n, 1);

    unsigned int seed = 123;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            seed = seed * 1103515245 + 12345;
            double v = ((double)((seed >> 16) & 0x7fff) / 16384.0) - 1.0;
            matrix_set(X, i, j, v);
        }
        /* Only feature 0 matters */
        double val = 5.0 * matrix_get(X, i, 0);
        matrix_set(y, i, 0, val);
    }

    LassoModel *m = lasso_model_create();
    m->alpha = 0.5;
    m->max_iter = 2000;
    m->tol = 1e-7;
    Estimator *e = lasso_fit((Estimator *)m, X, y);
    ASSERT_NOT_NULL(e);
    ASSERT_EQ(m->base.is_fitted, 1);

    /* Feature 0 should be significant */
    ASSERT(fabs(m->weights->data[0]) > 1.0);

    /* Features 1-3 should be small (pushed toward 0 by L1) */
    int zero_count = 0;
    for (int j = 1; j < p; j++) {
        if (fabs(m->weights->data[j]) < 0.5) zero_count++;
    }
    /* At least 2 of the 3 irrelevant features should be small */
    ASSERT(zero_count >= 2);

    /* R² should be decent */
    double r2 = lasso_score((Estimator *)m, X, y);
    ASSERT(r2 > 0.5);

    matrix_free(X);
    matrix_free(y);
    lasso_free((Estimator *)m);
}

TEST(test_lasso_zeroing) {
    /* Very high alpha should drive all weights to ~0 */
    int n = 30;
    int p = 3;
    Matrix *X = matrix_alloc(n, p);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n; i++) {
        matrix_set(X, i, 0, (double)i / 10.0);
        matrix_set(X, i, 1, (double)(i - 15) / 5.0);
        matrix_set(X, i, 2, cos((double)i));
        matrix_set(y, i, 0, 2.0 * matrix_get(X, i, 0));
    }

    LassoModel *m = lasso_model_create();
    m->alpha = 100.0;  /* very high */
    m->max_iter = 2000;
    Estimator *e = lasso_fit((Estimator *)m, X, y);
    ASSERT_NOT_NULL(e);

    /* All weights should be ~0 */
    for (int j = 0; j < p; j++) {
        ASSERT_NEAR(m->weights->data[j], 0.0, 0.1);
    }

    matrix_free(X);
    matrix_free(y);
    lasso_free((Estimator *)m);
}

int main(void) {
    printf("Lasso Regression Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_lasso_create_free);
    RUN_TEST(test_lasso_fit_predict);
    RUN_TEST(test_lasso_zeroing);

    TEST_SUMMARY();
}
