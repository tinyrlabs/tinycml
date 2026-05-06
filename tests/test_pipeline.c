/**
 * @file test_pipeline.c
 * @brief Unit tests for Pipeline (chaining transformers + estimators)
 */

#include "test_harness.h"
#include "matrix.h"
#include "pipeline.h"
#include "logistic_regression.h"
#include <math.h>

TEST(test_pipeline_create_free) {
    Pipeline *pipe = pipeline_create();
    ASSERT_NOT_NULL(pipe);
    ASSERT_EQ(pipe->n_steps, 0);
    ASSERT_EQ(pipe->base.is_fitted, 0);

    /* Add a transformer step */
    StandardScalerTransformer *scaler = standard_scaler_transformer_create();
    ASSERT_NOT_NULL(scaler);
    int rc = pipeline_add_transformer(pipe, "scaler", (Transformer*)scaler);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(pipe->n_steps, 1);

    pipeline_free((Estimator*)pipe);
}

TEST(test_pipeline_scaler_lr) {
    /* Create simple binary classification data: 2 clusters */
    int n = 30;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n / 2; i++) {
        /* Class 0: centered around (1, 1) */
        matrix_set(X, i, 0, 1.0 + 0.3 * (i % 3));
        matrix_set(X, i, 1, 1.0 + 0.2 * (i % 4));
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = n / 2; i < n; i++) {
        /* Class 1: centered around (5, 5) */
        matrix_set(X, i, 0, 5.0 + 0.3 * (i % 3));
        matrix_set(X, i, 1, 5.0 + 0.2 * (i % 4));
        matrix_set(y, i, 0, 1.0);
    }

    Pipeline *pipe = pipeline_create();
    ASSERT_NOT_NULL(pipe);

    /* Add StandardScaler transformer */
    StandardScalerTransformer *scaler = standard_scaler_transformer_create();
    ASSERT_NOT_NULL(scaler);
    pipeline_add_transformer(pipe, "scaler", (Transformer*)scaler);

    /* Add LogisticRegression estimator */
    LogisticRegressionModel *lr = logreg_model_create();
    ASSERT_NOT_NULL(lr);
    pipeline_add_estimator(pipe, "logreg", (Estimator*)lr);

    /* Fit pipeline */
    Estimator *fitted = pipeline_fit((Estimator*)pipe, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(pipe->base.is_fitted, 1);

    /* Predict */
    Matrix *preds = pipeline_predict((Estimator*)pipe, X);
    ASSERT_NOT_NULL(preds);
    ASSERT_EQ(preds->rows, (size_t)n);

    /* Compute accuracy */
    int correct = 0;
    for (int i = 0; i < n; i++) {
        double pred = matrix_get(preds, i, 0);
        double actual = matrix_get(y, i, 0);
        if (((int)(pred + 0.5)) == (int)actual) {
            correct++;
        }
    }
    double accuracy = (double)correct / n;
    ASSERT(accuracy > 0.6);

    matrix_free(preds);
    pipeline_free((Estimator*)pipe);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_pipeline_predict) {
    /* Simple test: pipeline with scaler + logreg, verify predict output format */
    int n = 10;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n / 2; i++) {
        matrix_set(X, i, 0, 0.0 + i * 0.1);
        matrix_set(X, i, 1, 0.0 + i * 0.1);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = n / 2; i < n; i++) {
        matrix_set(X, i, 0, 10.0 + (i - n / 2) * 0.1);
        matrix_set(X, i, 1, 10.0 + (i - n / 2) * 0.1);
        matrix_set(y, i, 0, 1.0);
    }

    Pipeline *pipe = pipeline_create();
    StandardScalerTransformer *scaler = standard_scaler_transformer_create();
    pipeline_add_transformer(pipe, "scaler", (Transformer*)scaler);

    LogisticRegressionModel *lr = logreg_model_create();
    pipeline_add_estimator(pipe, "logreg", (Estimator*)lr);

    pipeline_fit((Estimator*)pipe, X, y);

    Matrix *preds = pipeline_predict((Estimator*)pipe, X);
    ASSERT_NOT_NULL(preds);
    ASSERT_EQ(preds->rows, (size_t)n);
    ASSERT_EQ(preds->cols, 1);

    /* All predictions should be 0 or 1 */
    for (int i = 0; i < n; i++) {
        double val = matrix_get(preds, i, 0);
        ASSERT(val == 0.0 || val == 1.0);
    }

    matrix_free(preds);
    pipeline_free((Estimator*)pipe);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_pipeline_empty) {
    /* Empty pipeline: fit should return NULL */
    Pipeline *pipe = pipeline_create();
    ASSERT_NOT_NULL(pipe);

    Matrix *X = matrix_alloc(3, 2);
    Matrix *y = matrix_alloc(3, 1);
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 2.0);
    matrix_set(X, 1, 0, 3.0); matrix_set(X, 1, 1, 4.0);
    matrix_set(X, 2, 0, 5.0); matrix_set(X, 2, 1, 6.0);
    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 1.0);
    matrix_set(y, 2, 0, 0.0);

    Estimator *result = pipeline_fit((Estimator*)pipe, X, y);
    ASSERT_NULL(result);

    pipeline_free((Estimator*)pipe);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_pipeline_single_step) {
    /* Pipeline with just a classifier (no transformer) */
    int n = 20;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n / 2; i++) {
        matrix_set(X, i, 0, -5.0 + i * 0.1);
        matrix_set(X, i, 1, -5.0 + i * 0.1);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = n / 2; i < n; i++) {
        matrix_set(X, i, 0, 5.0 + (i - n / 2) * 0.1);
        matrix_set(X, i, 1, 5.0 + (i - n / 2) * 0.1);
        matrix_set(y, i, 0, 1.0);
    }

    Pipeline *pipe = pipeline_create();
    LogisticRegressionModel *lr = logreg_model_create();
    ASSERT_NOT_NULL(lr);
    pipeline_add_estimator(pipe, "logreg", (Estimator*)lr);

    Estimator *fitted = pipeline_fit((Estimator*)pipe, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(pipe->base.is_fitted, 1);

    Matrix *preds = pipeline_predict((Estimator*)pipe, X);
    ASSERT_NOT_NULL(preds);
    ASSERT_EQ(preds->rows, (size_t)n);

    matrix_free(preds);
    pipeline_free((Estimator*)pipe);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Pipeline Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_pipeline_create_free);
    RUN_TEST(test_pipeline_scaler_lr);
    RUN_TEST(test_pipeline_predict);
    RUN_TEST(test_pipeline_empty);
    RUN_TEST(test_pipeline_single_step);

    TEST_SUMMARY();
}
