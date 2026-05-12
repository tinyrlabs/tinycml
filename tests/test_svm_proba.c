/**
 * @file test_svm_proba.c
 * @brief Unit tests for SVM predict_proba
 */

#include "test_harness.h"
#include "matrix.h"
#include "svm.h"

TEST(test_svm_proba_range) {
    /* Train SVM on linearly separable data, verify probabilities in [0, 1] */
    int N = 20;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, -2.0 + (i % 5) * 0.3);
        matrix_set(X, i, 1, -2.0 + (i / 5) * 0.5);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10 + i, 0, 2.0 + (i % 5) * 0.3);
        matrix_set(X, 10 + i, 1, 2.0 + (i / 5) * 0.5);
        matrix_set(y, 10 + i, 0, 1.0);
    }

    LinearSVC *svc = linear_svc_create();
    ASSERT_NOT_NULL(svc);
    svc->learning_rate = 0.01;
    svc->max_iter = 2000;

    svc->base.fit((Estimator *)svc, X, y);
    ASSERT_EQ(svc->base.is_fitted, 1);

    /* Get probabilities via vtable */
    Matrix *proba = svc->base.predict_proba((const Estimator *)svc, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, 1);

    /* All probabilities should be in [0, 1] */
    for (int i = 0; i < N; i++) {
        ASSERT(proba->data[i] >= 0.0);
        ASSERT(proba->data[i] <= 1.0);
    }

    /* Also test direct function call */
    Matrix *proba2 = linear_svc_predict_proba(svc, X);
    ASSERT_NOT_NULL(proba2);
    ASSERT_NEAR(proba2->data[0], proba->data[0], 1e-10);

    matrix_free(proba2);
    matrix_free(proba);
    matrix_free(X);
    matrix_free(y);
    svc->base.free((Estimator *)svc);
}

TEST(test_svm_proba_binary) {
    /* Verify calibrated output: negative samples should have low P(y=1),
     * positive samples should have high P(y=1) */
    int N = 10;
    Matrix *X = matrix_alloc(N, 1);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Negative class: far left */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, -10.0 + i * 0.5);
        matrix_set(y, i, 0, 0.0);
    }
    /* Positive class: far right */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5 + i, 0, 10.0 + i * 0.5);
        matrix_set(y, 5 + i, 0, 1.0);
    }

    LinearSVC *svc = linear_svc_create();
    ASSERT_NOT_NULL(svc);
    svc->learning_rate = 0.01;
    svc->max_iter = 2000;

    svc->base.fit((Estimator *)svc, X, y);
    ASSERT_EQ(svc->base.is_fitted, 1);

    Matrix *proba = svc->base.predict_proba((const Estimator *)svc, X);
    ASSERT_NOT_NULL(proba);

    /* Negative samples: P(y=1) should be < 0.5 */
    for (int i = 0; i < 5; i++) {
        ASSERT(proba->data[i] < 0.5);
    }
    /* Positive samples: P(y=1) should be > 0.5 */
    for (int i = 5; i < 10; i++) {
        ASSERT(proba->data[i] > 0.5);
    }

    matrix_free(proba);
    matrix_free(X);
    matrix_free(y);
    svc->base.free((Estimator *)svc);
}

int main(void) {
    printf("SVM Predict Proba Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_svm_proba_range);
    RUN_TEST(test_svm_proba_binary);

    TEST_SUMMARY();
}
