/**
 * @file test_logreg.c
 * @brief Unit tests for logistic regression
 */

#include "test_harness.h"
#include "matrix.h"
#include "logistic_regression.h"

TEST(test_sigmoid) {
    /* sigmoid(0) = 0.5 */
    ASSERT_NEAR(sigmoid(0.0), 0.5, 1e-10);

    /* sigmoid(large positive) approaches 1 */
    ASSERT_NEAR(sigmoid(100.0), 1.0, 1e-10);

    /* sigmoid(large negative) approaches 0 */
    ASSERT_NEAR(sigmoid(-100.0), 0.0, 1e-10);

    /* sigmoid(1) = 1/(1+exp(-1)) */
    ASSERT_NEAR(sigmoid(1.0), 1.0 / (1.0 + exp(-1.0)), 1e-10);

    /* sigmoid(-1) = 1/(1+exp(1)) */
    ASSERT_NEAR(sigmoid(-1.0), 1.0 / (1.0 + exp(1.0)), 1e-10);
}

TEST(test_logreg_fit_and_predict) {
    /* Linearly separable data with bias column:
     * Class 0: far left, Class 1: far right
     * X = [[1, -5], [1, -3], [1, -1], [1, 3], [1, 5], [1, 7]]
     * y = [0, 0, 0, 1, 1, 1]
     */
    Matrix *X = matrix_alloc(6, 2);
    Matrix *y = matrix_alloc(6, 1);

    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, -5.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, -3.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1, -1.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1,  3.0);
    matrix_set(X, 4, 0, 1.0); matrix_set(X, 4, 1,  5.0);
    matrix_set(X, 5, 0, 1.0); matrix_set(X, 5, 1,  7.0);

    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 0.0);
    matrix_set(y, 2, 0, 0.0);
    matrix_set(y, 3, 0, 1.0);
    matrix_set(y, 4, 0, 1.0);
    matrix_set(y, 5, 0, 1.0);

    Matrix *weights = logreg_fit(X, y, 0.5, 500);
    ASSERT_NOT_NULL(weights);

    /* Predict class labels */
    Matrix *pred = logreg_predict(X, weights, 0.5);
    ASSERT_NOT_NULL(pred);

    /* Should perfectly classify training data */
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ((int)matrix_get(pred, i, 0), (int)matrix_get(y, i, 0));
    }

    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
    matrix_free(pred);
}

TEST(test_logreg_predict_proba) {
    /* Simple data with bias column */
    Matrix *X = matrix_alloc(4, 2);
    Matrix *y = matrix_alloc(4, 1);

    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, -10.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1,  -2.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1,   2.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1,  10.0);

    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 0.0);
    matrix_set(y, 2, 0, 1.0);
    matrix_set(y, 3, 0, 1.0);

    Matrix *weights = logreg_fit(X, y, 0.5, 500);
    ASSERT_NOT_NULL(weights);

    Matrix *proba = logreg_predict_proba(X, weights);
    ASSERT_NOT_NULL(proba);

    /* Probabilities should be between 0 and 1 */
    for (int i = 0; i < 4; i++) {
        double p = matrix_get(proba, i, 0);
        ASSERT(p >= 0.0 && p <= 1.0);
    }

    /* Low feature values should have low probability, high values high */
    ASSERT(matrix_get(proba, 0, 0) < matrix_get(proba, 3, 0));

    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
    matrix_free(proba);
}

TEST(test_logreg_null_inputs) {
    /* NULL inputs should return NULL */
    Matrix *X = matrix_alloc(2, 2);
    Matrix *y = matrix_alloc(2, 1);

    Matrix *result = logreg_fit(NULL, y, 0.1, 100);
    ASSERT_NULL(result);

    result = logreg_fit(X, NULL, 0.1, 100);
    ASSERT_NULL(result);

    result = logreg_predict_proba(NULL, X);
    ASSERT_NULL(result);

    result = logreg_predict(NULL, X, 0.5);
    ASSERT_NULL(result);

    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Logistic Regression Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_sigmoid);
    RUN_TEST(test_logreg_fit_and_predict);
    RUN_TEST(test_logreg_predict_proba);
    RUN_TEST(test_logreg_null_inputs);

    TEST_SUMMARY();
}
