/**
 * @file test_svm.c
 * @brief Unit tests for Linear SVM classifier
 */

#include "test_harness.h"
#include "matrix.h"
#include "svm.h"

TEST(test_svm_create_free) {
    LinearSVC *svc = linear_svc_create();
    ASSERT_NOT_NULL(svc);
    ASSERT_EQ(svc->base.type, MODEL_SVM);
    ASSERT_EQ(svc->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(svc->base.is_fitted, 0);
    ASSERT_NEAR(svc->C, 1.0, 1e-9);
    ASSERT_NEAR(svc->learning_rate, 0.001, 1e-9);
    ASSERT_EQ(svc->max_iter, 1000);
    ASSERT_NEAR(svc->tol, 1e-4, 1e-9);
    ASSERT_NULL(svc->weights);
    ASSERT_EQ(svc->n_features, 0);

    svc->base.free((Estimator *)svc);
}

TEST(test_svm_binary) {
    /* Linearly separable binary data with labels {0, 1} */
    int N = 20;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0: points in lower-left region */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, -2.0 + (i % 5) * 0.3);
        matrix_set(X, i, 1, -2.0 + (i / 5) * 0.5);
        matrix_set(y, i, 0, 0.0);
    }
    /* Class 1: points in upper-right region */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10 + i, 0, 2.0 + (i % 5) * 0.3);
        matrix_set(X, 10 + i, 1, 2.0 + (i / 5) * 0.5);
        matrix_set(y, 10 + i, 0, 1.0);
    }

    LinearSVC *svc = linear_svc_create();
    ASSERT_NOT_NULL(svc);

    /* Increase learning rate and iterations for better convergence */
    svc->learning_rate = 0.01;
    svc->max_iter = 2000;

    Estimator *fitted = svc->base.fit((Estimator *)svc, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(svc->base.is_fitted, 1);
    ASSERT_EQ(svc->n_features, 2);
    ASSERT_NOT_NULL(svc->weights);

    /* Accuracy should be high on this separable data */
    double acc = svc->base.score((const Estimator *)svc, X, y);
    ASSERT(acc > 0.8);

    matrix_free(X);
    matrix_free(y);
    svc->base.free((Estimator *)svc);
}

TEST(test_svm_linear) {
    /* Simple 1D linearly separable data with labels {-1, +1} */
    int N = 10;
    Matrix *X = matrix_alloc(N, 1);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Negative class */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, -5.0 + i * 0.5);
        matrix_set(y, i, 0, -1.0);
    }
    /* Positive class */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5 + i, 0, 5.0 + i * 0.5);
        matrix_set(y, 5 + i, 0, 1.0);
    }

    LinearSVC *svc = linear_svc_create();
    ASSERT_NOT_NULL(svc);
    svc->learning_rate = 0.01;
    svc->max_iter = 2000;

    svc->base.fit((Estimator *)svc, X, y);
    ASSERT_EQ(svc->base.is_fitted, 1);

    /* Predict on training data */
    Matrix *pred = svc->base.predict((const Estimator *)svc, X);
    ASSERT_NOT_NULL(pred);

    /* Should perfectly separate */
    double acc = svc->base.score((const Estimator *)svc, X, y);
    ASSERT(acc > 0.9);

    /* Test a new sample: negative */
    Matrix *X_new = matrix_alloc(2, 1);
    matrix_set(X_new, 0, 0, -10.0);   /* clearly negative */
    matrix_set(X_new, 1, 0,  10.0);   /* clearly positive */

    Matrix *pred_new = svc->base.predict((const Estimator *)svc, X_new);
    ASSERT_NOT_NULL(pred_new);
    ASSERT(pred_new->data[0] < 0.0);
    ASSERT(pred_new->data[1] > 0.0);

    matrix_free(pred_new);
    matrix_free(X_new);
    matrix_free(pred);
    matrix_free(X);
    matrix_free(y);
    svc->base.free((Estimator *)svc);
}

int main(void) {
    RUN_TEST(test_svm_create_free);
    RUN_TEST(test_svm_binary);
    RUN_TEST(test_svm_linear);
    TEST_SUMMARY();
}
