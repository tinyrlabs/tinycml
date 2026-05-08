/**
 * @file test_feature_selection.c
 * @brief Unit tests for feature selection (SelectKBest, VarianceThreshold)
 */

#include "test_harness.h"
#include "matrix.h"
#include "feature_selection.h"
#include <math.h>

TEST(test_select_k_best_create_free) {
    SelectKBest *skb = select_k_best_create(SCORE_F_REGRESSION, 3);
    ASSERT_NOT_NULL(skb);
    ASSERT_EQ(skb->score_func, SCORE_F_REGRESSION);
    ASSERT_EQ(skb->k, 3);
    ASSERT_EQ(skb->base.type, MODEL_FEATURE_SELECTOR);
    ASSERT_EQ(skb->base.task, TASK_TRANSFORMATION);
    ASSERT_EQ(skb->base.is_fitted, 0);

    select_k_best_free((Estimator*)skb);
}

TEST(test_select_k_best_fit_transform) {
    /* 20 samples, 5 features: 2 informative (correlated with y), 3 noise */
    int n_samples = 20;
    int n_features = 5;
    Matrix *X = matrix_alloc(n_samples, n_features);
    Matrix *y = matrix_alloc(n_samples, 1);

    for (int i = 0; i < n_samples; i++) {
        double x0 = (double)(i % 5);
        double x1 = (double)(i / 5);
        /* Feature 0: informative (strong correlation with y) */
        matrix_set(X, i, 0, x0 * 10.0);
        /* Feature 1: noise */
        matrix_set(X, i, 1, (double)((i * 7 + 3) % 11) * 0.1);
        /* Feature 2: informative (linear with y) */
        matrix_set(X, i, 2, x1 * 5.0);
        /* Feature 3: noise */
        matrix_set(X, i, 3, (double)((i * 13 + 1) % 7) * 0.01);
        /* Feature 4: noise */
        matrix_set(X, i, 4, 1.0);  /* Constant = zero correlation */
        /* Target: linear combination of features 0 and 2 */
        matrix_set(y, i, 0, x0 * 10.0 + x1 * 5.0);
    }

    SelectKBest *skb = select_k_best_create(SCORE_F_REGRESSION, 2);
    ASSERT_NOT_NULL(skb);

    Estimator *fitted = select_k_best_fit((Estimator*)skb, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(skb->base.is_fitted, 1);
    ASSERT_EQ(skb->n_features_selected_, 2);

    Matrix *X_selected = select_k_best_transform((Estimator*)skb, X);
    ASSERT_NOT_NULL(X_selected);
    ASSERT_EQ(X_selected->rows, n_samples);
    ASSERT_EQ(X_selected->cols, 2);

    matrix_free(X_selected);
    select_k_best_free((Estimator*)skb);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_variance_threshold_create_free) {
    VarianceThreshold *vt = variance_threshold_create(0.01);
    ASSERT_NOT_NULL(vt);
    ASSERT_NEAR(vt->threshold, 0.01, 1e-9);
    ASSERT_EQ(vt->base.type, MODEL_FEATURE_SELECTOR);
    ASSERT_EQ(vt->base.task, TASK_TRANSFORMATION);
    ASSERT_EQ(vt->base.is_fitted, 0);

    variance_threshold_free((Estimator*)vt);
}

TEST(test_variance_threshold_fit_transform) {
    /* 5 samples, 3 features: feature 1 has zero variance */
    Matrix *X = matrix_alloc(5, 3);
    Matrix *y = matrix_alloc(5, 1);

    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, (double)(i + 1));       /* Varying: var > 0 */
        matrix_set(X, i, 1, 3.0);                    /* Constant: var = 0 */
        matrix_set(X, i, 2, (double)((i + 1) * 2)); /* Varying: var > 0 */
        matrix_set(y, i, 0, (double)i);
    }

    VarianceThreshold *vt = variance_threshold_create(0.0);
    ASSERT_NOT_NULL(vt);

    Estimator *fitted = variance_threshold_fit((Estimator*)vt, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(vt->base.is_fitted, 1);
    ASSERT_EQ(vt->n_features_selected_, 2);  /* Feature 1 removed */

    /* Verify variances */
    const double *variances = variance_threshold_variances(vt);
    ASSERT_NOT_NULL(variances);
    ASSERT(variances[0] > 0.0);
    ASSERT_NEAR(variances[1], 0.0, 1e-12);
    ASSERT(variances[2] > 0.0);

    /* Verify support mask */
    const int *support = variance_threshold_get_support(vt);
    ASSERT_EQ(support[0], 1);
    ASSERT_EQ(support[1], 0);
    ASSERT_EQ(support[2], 1);

    Matrix *X_filtered = variance_threshold_transform((Estimator*)vt, X);
    ASSERT_NOT_NULL(X_filtered);
    ASSERT_EQ(X_filtered->rows, 5);
    ASSERT_EQ(X_filtered->cols, 2);

    /* Verify data correctness: columns 0 and 2 kept */
    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(matrix_get(X_filtered, i, 0), (double)(i + 1), 1e-9);
        ASSERT_NEAR(matrix_get(X_filtered, i, 1), (double)((i + 1) * 2), 1e-9);
    }

    matrix_free(X_filtered);
    variance_threshold_free((Estimator*)vt);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_variance_threshold_keeps_all) {
    /* All features have high variance, all should be kept */
    Matrix *X = matrix_alloc(6, 3);
    Matrix *y = matrix_alloc(6, 1);

    for (int i = 0; i < 6; i++) {
        matrix_set(X, i, 0, (double)(i * 10));
        matrix_set(X, i, 1, (double)(i * 100));
        matrix_set(X, i, 2, (double)(i * 50));
        matrix_set(y, i, 0, (double)i);
    }

    VarianceThreshold *vt = variance_threshold_create(0.01);
    ASSERT_NOT_NULL(vt);

    Estimator *fitted = variance_threshold_fit((Estimator*)vt, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(vt->n_features_selected_, 3);

    Matrix *X_filtered = variance_threshold_transform((Estimator*)vt, X);
    ASSERT_NOT_NULL(X_filtered);
    ASSERT_EQ(X_filtered->rows, 6);
    ASSERT_EQ(X_filtered->cols, 3);

    matrix_free(X_filtered);
    variance_threshold_free((Estimator*)vt);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Feature Selection Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_select_k_best_create_free);
    RUN_TEST(test_select_k_best_fit_transform);
    RUN_TEST(test_variance_threshold_create_free);
    RUN_TEST(test_variance_threshold_fit_transform);
    RUN_TEST(test_variance_threshold_keeps_all);

    TEST_SUMMARY();
}
