/**
 * @file test_onehot.c
 * @brief Unit tests for OneHotEncoder
 */

#include "test_harness.h"
#include "matrix.h"
#include "onehot_encoder.h"

TEST(test_onehot_create_free) {
    OneHotEncoder *enc = onehot_encoder_create();
    ASSERT_NOT_NULL(enc);
    ASSERT_EQ(enc->is_fitted, 0);
    ASSERT_EQ(enc->n_features, 0);
    ASSERT_EQ(enc->total_output_cols, 0);
    ASSERT_NULL(enc->n_categories);
    ASSERT_NULL(enc->categories_);

    onehot_encoder_free(enc);
}

TEST(test_onehot_fit_transform) {
    /* 3 samples, 1 feature with categories [0, 1, 2] */
    Matrix *X = matrix_alloc(3, 1);
    ASSERT_NOT_NULL(X);
    matrix_set(X, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0);
    matrix_set(X, 2, 0, 2.0);

    OneHotEncoder *enc = onehot_encoder_create();
    ASSERT_NOT_NULL(enc);

    int ret = onehot_encoder_fit(enc, X);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(enc->is_fitted, 1);
    ASSERT_EQ(enc->n_features, 1);
    ASSERT_EQ(enc->n_categories[0], 3);
    ASSERT_EQ(enc->total_output_cols, 3);

    Matrix *result = onehot_encoder_transform(enc, X);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ(result->rows, 3);
    ASSERT_EQ(result->cols, 3);

    /* Row 0: [1, 0, 0] for category 0 */
    ASSERT_NEAR(matrix_get(result, 0, 0), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 0, 1), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 0, 2), 0.0, 1e-9);

    /* Row 1: [0, 1, 0] for category 1 */
    ASSERT_NEAR(matrix_get(result, 1, 0), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 1, 1), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 1, 2), 0.0, 1e-9);

    /* Row 2: [0, 0, 1] for category 2 */
    ASSERT_NEAR(matrix_get(result, 2, 0), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 2, 1), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 2, 2), 1.0, 1e-9);

    matrix_free(result);
    onehot_encoder_free(enc);
    matrix_free(X);
}

TEST(test_onehot_binary) {
    /* 4 samples, 2 binary features */
    Matrix *X = matrix_alloc(4, 2);
    ASSERT_NOT_NULL(X);
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 0.0); matrix_set(X, 1, 1, 1.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1, 0.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 1.0);

    OneHotEncoder *enc = onehot_encoder_create();
    ASSERT_NOT_NULL(enc);

    int ret = onehot_encoder_fit(enc, X);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(enc->is_fitted, 1);
    ASSERT_EQ(enc->n_features, 2);
    ASSERT_EQ(enc->n_categories[0], 2);
    ASSERT_EQ(enc->n_categories[1], 2);
    ASSERT_EQ(enc->total_output_cols, 4);

    Matrix *result = onehot_encoder_transform(enc, X);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ(result->rows, 4);
    ASSERT_EQ(result->cols, 4);

    /* Row 0: [1,0, 1,0] — feature0=0, feature1=0 */
    ASSERT_NEAR(matrix_get(result, 0, 0), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 0, 1), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 0, 2), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 0, 3), 0.0, 1e-9);

    /* Row 1: [1,0, 0,1] — feature0=0, feature1=1 */
    ASSERT_NEAR(matrix_get(result, 1, 0), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 1, 1), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 1, 2), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 1, 3), 1.0, 1e-9);

    /* Row 2: [0,1, 1,0] — feature0=1, feature1=0 */
    ASSERT_NEAR(matrix_get(result, 2, 0), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 2, 1), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 2, 2), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 2, 3), 0.0, 1e-9);

    /* Row 3: [0,1, 0,1] — feature0=1, feature1=1 */
    ASSERT_NEAR(matrix_get(result, 3, 0), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 3, 1), 1.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 3, 2), 0.0, 1e-9);
    ASSERT_NEAR(matrix_get(result, 3, 3), 1.0, 1e-9);

    matrix_free(result);
    onehot_encoder_free(enc);
    matrix_free(X);
}

int main(void) {
    RUN_TEST(test_onehot_create_free);
    RUN_TEST(test_onehot_fit_transform);
    RUN_TEST(test_onehot_binary);
    TEST_SUMMARY();
}
