/**
 * @file test_svm_rbf.c
 * @brief Unit tests for SVM with RBF kernel
 */

#include "test_harness.h"
#include "matrix.h"
#include "svm.h"

TEST(test_svm_rbf_create_free) {
    SVMClassifier *svm = svm_classifier_create(CML_KERNEL_RBF);
    ASSERT_NOT_NULL(svm);
    ASSERT_EQ(svm->base.type, MODEL_SVM);
    ASSERT_EQ(svm->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(svm->base.is_fitted, 0);
    ASSERT_EQ(svm->kernel, CML_KERNEL_RBF);
    ASSERT_NEAR(svm->C, 1.0, 1e-9);
    ASSERT_NEAR(svm->gamma, -1.0, 1e-9);
    ASSERT_NEAR(svm->lr, 0.01, 1e-9);
    ASSERT_EQ(svm->max_iter, 1000);
    ASSERT_NULL(svm->weights);
    ASSERT_NULL(svm->support_vectors);
    ASSERT_NULL(svm->alphas);
    ASSERT_EQ(svm->n_support, 0);
    ASSERT_EQ(svm->n_features, 0);

    svm->base.free((Estimator *)svm);
}

TEST(test_svm_rbf_xor) {
    /* XOR pattern: linear kernel should fail, RBF should succeed */
    /* (0,0)->+1, (1,1)->+1, (0,1)->-1, (1,0)->-1 */
    /* We add slight noise to make it more realistic */
    int N = 40;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Use deterministic seed for test data via matrix_set */
    double data[4][3] = {
        {0.0, 0.0,  1.0},
        {0.0, 1.0, -1.0},
        {1.0, 0.0, -1.0},
        {1.0, 1.0,  1.0}
    };

    /* Create 10 samples per XOR quadrant with small perturbations */
    for (int q = 0; q < 4; q++) {
        for (int i = 0; i < 10; i++) {
            int idx = q * 10 + i;
            /* Small deterministic offsets */
            double dx = (double)(i % 5) * 0.05 - 0.1;
            double dy = (double)(i / 5) * 0.05 - 0.05;
            matrix_set(X, idx, 0, data[q][0] + dx);
            matrix_set(X, idx, 1, data[q][1] + dy);
            matrix_set(y, idx, 0, data[q][2]);
        }
    }

    SVMClassifier *svm = svm_classifier_create(CML_KERNEL_RBF);
    ASSERT_NOT_NULL(svm);
    svm->gamma = 1.0;
    svm->lr = 0.05;
    svm->C = 10.0;
    svm->max_iter = 2000;

    Estimator *fitted = svm->base.fit((Estimator *)svm, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(svm->base.is_fitted, 1);
    ASSERT(svm->n_support > 0);

    /* Accuracy should be > 0.7 for XOR */
    double acc = svm->base.score((const Estimator *)svm, X, y);
    ASSERT(acc > 0.7);

    matrix_free(X);
    matrix_free(y);
    svm->base.free((Estimator *)svm);
}

TEST(test_svm_rbf_linear) {
    /* Linearly separable data with RBF kernel — should also work well */
    int N = 20;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class -1: lower-left region */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, -3.0 + (i % 5) * 0.3);
        matrix_set(X, i, 1, -3.0 + (i / 5) * 0.5);
        matrix_set(y, i, 0, -1.0);
    }
    /* Class +1: upper-right region */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10 + i, 0, 3.0 + (i % 5) * 0.3);
        matrix_set(X, 10 + i, 1, 3.0 + (i / 5) * 0.5);
        matrix_set(y, 10 + i, 0, 1.0);
    }

    SVMClassifier *svm = svm_classifier_create(CML_KERNEL_RBF);
    ASSERT_NOT_NULL(svm);
    svm->gamma = 0.5;
    svm->lr = 0.01;
    svm->C = 1.0;
    svm->max_iter = 1000;

    Estimator *fitted = svm->base.fit((Estimator *)svm, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(svm->base.is_fitted, 1);

    /* Accuracy should be > 0.8 */
    double acc = svm->base.score((const Estimator *)svm, X, y);
    ASSERT(acc > 0.8);

    matrix_free(X);
    matrix_free(y);
    svm->base.free((Estimator *)svm);
}

int main(void) {
    RUN_TEST(test_svm_rbf_create_free);
    RUN_TEST(test_svm_rbf_xor);
    RUN_TEST(test_svm_rbf_linear);
    TEST_SUMMARY();
}
