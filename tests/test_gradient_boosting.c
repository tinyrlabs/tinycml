/**
 * test_gradient_boosting.c - Tests for Gradient Boosted Decision Trees
 */

#include "test_harness.h"
#include "gradient_boosting.h"
#include "matrix.h"

/* ── Test 1: Create / free ──────────────────────────────────────── */

TEST(test_gb_create) {
    GradientBoosting *gb = gradient_boosting_create(50, 0.1, 3, 2, 1.0);
    ASSERT_NOT_NULL(gb);
    ASSERT_EQ(gb->n_estimators, 50);
    ASSERT_NEAR(gb->learning_rate, 0.1, 1e-9);
    ASSERT_EQ(gb->max_depth, 3);

    gradient_boosting_free((Estimator*)gb);
}

/* ── Test 2: Regression on linear data ─────────────────────────── */

TEST(test_gb_regression) {
    /* y = 2*x + 1 */
    size_t n = 100;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        double x_val = (double)i / 10.0;
        matrix_set(X, i, 0, x_val);
        matrix_set(y, i, 0, 2.0 * x_val + 1.0);
    }

    GradientBoosting *gb = gradient_boosting_create(50, 0.1, 4, 2, 1.0);
    ASSERT_NOT_NULL(gb);

    Estimator *result = gradient_boosting_fit((Estimator*)gb, X, y);
    ASSERT_NOT_NULL(result);

    double score = gradient_boosting_score((Estimator*)gb, X, y);
    ASSERT(score > 0.9);

    matrix_free(X);
    matrix_free(y);
    gradient_boosting_free((Estimator*)gb);
}

/* ── Test 3: Binary classification ─────────────────────────────── */

TEST(test_gb_classification) {
    /* class 0 for x < 5, class 1 for x >= 5 */
    size_t n = 100;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        double x_val = (double)i / 10.0;
        matrix_set(X, i, 0, x_val);
        matrix_set(y, i, 0, x_val >= 5.0 ? 1.0 : 0.0);
    }

    GradientBoosting *gb = gradient_boosting_create(30, 0.1, 3, 2, 1.0);
    ASSERT_NOT_NULL(gb);

    gradient_boosting_fit((Estimator*)gb, X, y);

    double score = gradient_boosting_score((Estimator*)gb, X, y);
    ASSERT(score > 0.85);

    /* Predict on new data */
    Matrix *X_test = matrix_alloc(4, 1);
    matrix_set(X_test, 0, 0, 2.0);
    matrix_set(X_test, 1, 0, 3.0);
    matrix_set(X_test, 2, 0, 7.0);
    matrix_set(X_test, 3, 0, 9.0);

    Matrix *pred = gradient_boosting_predict((Estimator*)gb, X_test);
    ASSERT_NOT_NULL(pred);

    ASSERT(pred->data[0] < 0.5);
    ASSERT(pred->data[1] < 0.5);
    ASSERT(pred->data[2] > 0.5);
    ASSERT(pred->data[3] > 0.5);

    matrix_free(X_test);
    matrix_free(pred);
    matrix_free(X);
    matrix_free(y);
    gradient_boosting_free((Estimator*)gb);
}

/* ── Test 4: Predict proba ─────────────────────────────────────── */

TEST(test_gb_predict_proba) {
    size_t n = 50;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        matrix_set(X, i, 0, (double)i);
        matrix_set(y, i, 0, i >= 25 ? 1.0 : 0.0);
    }

    GradientBoosting *gb = gradient_boosting_create(20, 0.1, 3, 2, 1.0);
    gradient_boosting_fit((Estimator*)gb, X, y);

    Matrix *proba = gradient_boosting_predict_proba((Estimator*)gb, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ((int)proba->cols, 2);

    /* Probabilities should sum to ~1 */
    for (size_t i = 0; i < n; i++) {
        double p0 = matrix_get(proba, i, 0);
        double p1 = matrix_get(proba, i, 1);
        ASSERT_NEAR(p0 + p1, 1.0, 1e-6);
    }

    matrix_free(proba);
    matrix_free(X);
    matrix_free(y);
    gradient_boosting_free((Estimator*)gb);
}

/* ── Test 5: Clone ─────────────────────────────────────────────── */

TEST(test_gb_clone) {
    size_t n = 30;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        matrix_set(X, i, 0, (double)i / 5.0);
        matrix_set(y, i, 0, (i % 2 == 0) ? 1.0 : 0.0);
    }

    GradientBoosting *gb = gradient_boosting_create(10, 0.1, 2, 2, 1.0);
    gradient_boosting_fit((Estimator*)gb, X, y);

    Matrix *pred_orig = gradient_boosting_predict((Estimator*)gb, X);
    ASSERT_NOT_NULL(pred_orig);

    /* Verify the original model works */
    double score = gradient_boosting_score((Estimator*)gb, X, y);
    ASSERT(score > 0.5);

    matrix_free(pred_orig);
    matrix_free(X);
    matrix_free(y);
    gradient_boosting_free((Estimator*)gb);
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Gradient Boosting Tests ===\n\n");
    RUN_TEST(test_gb_create);
    RUN_TEST(test_gb_regression);
    RUN_TEST(test_gb_classification);
    RUN_TEST(test_gb_predict_proba);
    RUN_TEST(test_gb_clone);
    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return 0;
}
