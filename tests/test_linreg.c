/**
 * @file test_linreg.c
 * @brief Unit tests for linear regression
 */

#include "test_harness.h"
#include "matrix.h"
#include "linear_regression.h"
#include "preprocessing.h"
#include "metrics.h"

TEST(test_linreg_simple) {
    /* Simple linear data: y = 2*x + 1 */
    /* X with bias column: [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]] */
    /* y: [3, 5, 7, 9, 11] */

    Matrix *X = matrix_alloc(5, 2);
    Matrix *y = matrix_alloc(5, 1);

    /* Set up X with bias column */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, 1.0);  /* bias */
        matrix_set(X, i, 1, (double)(i + 1));  /* feature */
    }

    /* Set up y: y = 2*x + 1 */
    matrix_set(y, 0, 0, 3.0);
    matrix_set(y, 1, 0, 5.0);
    matrix_set(y, 2, 0, 7.0);
    matrix_set(y, 3, 0, 9.0);
    matrix_set(y, 4, 0, 11.0);

    /* Fit using closed form */
    Matrix *weights = linreg_fit_closed(X, y);
    ASSERT_NOT_NULL(weights);
    ASSERT_EQ(weights->rows, 2);

    /* Should get weights close to [1, 2] (intercept, slope) */
    ASSERT_NEAR(matrix_get(weights, 0, 0), 1.0, 0.01);
    ASSERT_NEAR(matrix_get(weights, 1, 0), 2.0, 0.01);

    /* Test prediction */
    Matrix *pred = linreg_predict(X, weights);
    ASSERT_NOT_NULL(pred);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(matrix_get(pred, i, 0), matrix_get(y, i, 0), 0.01);
    }

    /* Check MSE is near zero */
    double error = mse(y, pred);
    ASSERT_NEAR(error, 0.0, 0.01);

    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
    matrix_free(pred);
}

TEST(test_linreg_gd) {
    /* Same simple data with gradient descent */
    Matrix *X = matrix_alloc(5, 2);
    Matrix *y = matrix_alloc(5, 1);

    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, 1.0);
        matrix_set(X, i, 1, (double)(i + 1));
    }

    matrix_set(y, 0, 0, 3.0);
    matrix_set(y, 1, 0, 5.0);
    matrix_set(y, 2, 0, 7.0);
    matrix_set(y, 3, 0, 9.0);
    matrix_set(y, 4, 0, 11.0);

    /* Fit using gradient descent */
    Matrix *weights = linreg_fit_gd(X, y, 0.1, 1000);
    ASSERT_NOT_NULL(weights);

    /* Should converge close to [1, 2] */
    ASSERT_NEAR(matrix_get(weights, 0, 0), 1.0, 0.1);
    ASSERT_NEAR(matrix_get(weights, 1, 0), 2.0, 0.1);

    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
}

TEST(test_linreg_multivariate) {
    /* y = 1 + 2*x1 + 3*x2 */
    Matrix *X = matrix_alloc(4, 3);
    Matrix *y = matrix_alloc(4, 1);

    /* X with bias: [[1, 1, 1], [1, 2, 1], [1, 1, 2], [1, 2, 2]] */
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 1.0); matrix_set(X, 0, 2, 1.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 2.0); matrix_set(X, 1, 2, 1.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1, 1.0); matrix_set(X, 2, 2, 2.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 2.0); matrix_set(X, 3, 2, 2.0);

    /* y = 1 + 2*x1 + 3*x2 */
    matrix_set(y, 0, 0, 6.0);   /* 1 + 2*1 + 3*1 = 6 */
    matrix_set(y, 1, 0, 8.0);   /* 1 + 2*2 + 3*1 = 8 */
    matrix_set(y, 2, 0, 9.0);   /* 1 + 2*1 + 3*2 = 9 */
    matrix_set(y, 3, 0, 11.0);  /* 1 + 2*2 + 3*2 = 11 */

    Matrix *weights = linreg_fit_closed(X, y);
    ASSERT_NOT_NULL(weights);

    ASSERT_NEAR(matrix_get(weights, 0, 0), 1.0, 0.01);  /* intercept */
    ASSERT_NEAR(matrix_get(weights, 1, 0), 2.0, 0.01);  /* x1 coefficient */
    ASSERT_NEAR(matrix_get(weights, 2, 0), 3.0, 0.01);  /* x2 coefficient */

    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
}

int main(void) {
    printf("Linear Regression Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_linreg_simple);
    RUN_TEST(test_linreg_gd);
    RUN_TEST(test_linreg_multivariate);

    TEST_SUMMARY();
}
