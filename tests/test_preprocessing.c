/**
 * @file test_preprocessing.c
 * @brief Unit tests for preprocessing functions
 */

#include "test_harness.h"
#include "matrix.h"
#include "preprocessing.h"
#include <math.h>

TEST(test_train_test_split) {
    /* Create 10 samples, 2 features */
    Matrix *X = matrix_alloc(10, 2);
    Matrix *y = matrix_alloc(10, 1);

    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, (double)i);
        matrix_set(X, i, 1, (double)(i * 2));
        matrix_set(y, i, 0, (double)(i % 2));
    }

    TrainTestSplit split = train_test_split(X, y, 0.2, 42);

    ASSERT_NOT_NULL(split.X_train);
    ASSERT_NOT_NULL(split.X_test);
    ASSERT_NOT_NULL(split.y_train);
    ASSERT_NOT_NULL(split.y_test);

    /* With 10 samples and 0.2 test ratio: 8 train, 2 test */
    ASSERT_EQ(split.X_train->rows, 8);
    ASSERT_EQ(split.X_test->rows, 2);
    ASSERT_EQ(split.y_train->rows, 8);
    ASSERT_EQ(split.y_test->rows, 2);

    /* Total samples preserved */
    ASSERT_EQ(split.X_train->rows + split.X_test->rows, 10);
    ASSERT_EQ(split.y_train->rows + split.y_test->rows, 10);

    /* Feature count preserved */
    ASSERT_EQ(split.X_train->cols, 2);
    ASSERT_EQ(split.X_test->cols, 2);

    train_test_split_free(&split);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_standardize) {
    /* Create data with known mean and std */
    Matrix *X = matrix_alloc(5, 2);

    /* Column 0: [1, 2, 3, 4, 5] => mean=3, std~1.414 */
    /* Column 1: [10, 20, 30, 40, 50] => mean=30, std~14.14 */
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 10.0);
    matrix_set(X, 1, 0, 2.0); matrix_set(X, 1, 1, 20.0);
    matrix_set(X, 2, 0, 3.0); matrix_set(X, 2, 1, 30.0);
    matrix_set(X, 3, 0, 4.0); matrix_set(X, 3, 1, 40.0);
    matrix_set(X, 4, 0, 5.0); matrix_set(X, 4, 1, 50.0);

    Scaler *scaler = standardize_fit(X);
    ASSERT_NOT_NULL(scaler);
    ASSERT_EQ(scaler->n_features, 2);

    /* Check fitted means */
    ASSERT_NEAR(scaler->means[0], 3.0, 0.01);
    ASSERT_NEAR(scaler->means[1], 30.0, 0.01);

    Matrix *X_scaled = standardize_transform(X, scaler);
    ASSERT_NOT_NULL(X_scaled);

    /* Verify mean ~ 0 for each column */
    for (size_t j = 0; j < 2; j++) {
        double col_mean = 0.0;
        for (size_t i = 0; i < X_scaled->rows; i++) {
            col_mean += matrix_get(X_scaled, i, j);
        }
        col_mean /= (double)X_scaled->rows;
        ASSERT_NEAR(col_mean, 0.0, 0.01);
    }

    /* Verify std ~ 1 for each column */
    for (size_t j = 0; j < 2; j++) {
        double col_mean = 0.0;
        for (size_t i = 0; i < X_scaled->rows; i++) {
            col_mean += matrix_get(X_scaled, i, j);
        }
        col_mean /= (double)X_scaled->rows;

        double variance = 0.0;
        for (size_t i = 0; i < X_scaled->rows; i++) {
            double diff = matrix_get(X_scaled, i, j) - col_mean;
            variance += diff * diff;
        }
        variance /= (double)X_scaled->rows;
        double col_std = sqrt(variance);
        ASSERT_NEAR(col_std, 1.0, 0.05);
    }

    matrix_free(X_scaled);
    scaler_free(scaler);
    matrix_free(X);
}

TEST(test_minmax_scale) {
    /* Create data with known min/max */
    Matrix *X = matrix_alloc(5, 2);

    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 100.0);
    matrix_set(X, 1, 0, 2.0); matrix_set(X, 1, 1, 200.0);
    matrix_set(X, 2, 0, 3.0); matrix_set(X, 2, 1, 300.0);
    matrix_set(X, 3, 0, 4.0); matrix_set(X, 3, 1, 400.0);
    matrix_set(X, 4, 0, 5.0); matrix_set(X, 4, 1, 500.0);

    MinMaxScaler *scaler = minmax_fit(X);
    ASSERT_NOT_NULL(scaler);
    ASSERT_EQ(scaler->n_features, 2);

    /* Check fitted min/max */
    ASSERT_NEAR(scaler->mins[0], 1.0, 0.01);
    ASSERT_NEAR(scaler->maxs[0], 5.0, 0.01);
    ASSERT_NEAR(scaler->mins[1], 100.0, 0.01);
    ASSERT_NEAR(scaler->maxs[1], 500.0, 0.01);

    Matrix *X_scaled = minmax_transform(X, scaler);
    ASSERT_NOT_NULL(X_scaled);

    /* Verify min=0 and max=1 for each column */
    for (size_t j = 0; j < 2; j++) {
        double col_min = 1e9, col_max = -1e9;
        for (size_t i = 0; i < X_scaled->rows; i++) {
            double v = matrix_get(X_scaled, i, j);
            if (v < col_min) col_min = v;
            if (v > col_max) col_max = v;
        }
        ASSERT_NEAR(col_min, 0.0, 0.01);
        ASSERT_NEAR(col_max, 1.0, 0.01);
    }

    /* All values should be in [0, 1] */
    for (size_t i = 0; i < X_scaled->rows; i++) {
        for (size_t j = 0; j < X_scaled->cols; j++) {
            double v = matrix_get(X_scaled, i, j);
            ASSERT(v >= -0.001 && v <= 1.001);
        }
    }

    matrix_free(X_scaled);
    minmax_scaler_free(scaler);
    matrix_free(X);
}

TEST(test_scaler_free) {
    /* Create scaler, fit, free — no crash */
    Matrix *X = matrix_alloc(3, 2);
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 4.0);
    matrix_set(X, 1, 0, 2.0); matrix_set(X, 1, 1, 5.0);
    matrix_set(X, 2, 0, 3.0); matrix_set(X, 2, 1, 6.0);

    /* Standard scaler */
    Scaler *scaler = standardize_fit(X);
    ASSERT_NOT_NULL(scaler);
    scaler_free(scaler);

    /* MinMax scaler */
    MinMaxScaler *mm_scaler = minmax_fit(X);
    ASSERT_NOT_NULL(mm_scaler);
    minmax_scaler_free(mm_scaler);

    /* Fit and transform then free */
    Scaler *scaler2 = NULL;
    Matrix *X_scaled = standardize_fit_transform(X, &scaler2);
    ASSERT_NOT_NULL(X_scaled);
    ASSERT_NOT_NULL(scaler2);

    scaler_free(scaler2);
    matrix_free(X_scaled);
    matrix_free(X);
}

TEST(test_add_bias_column) {
    Matrix *X = matrix_alloc(3, 2);
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 2.0);
    matrix_set(X, 1, 0, 3.0); matrix_set(X, 1, 1, 4.0);
    matrix_set(X, 2, 0, 5.0); matrix_set(X, 2, 1, 6.0);

    Matrix *X_bias = add_bias_column(X);
    ASSERT_NOT_NULL(X_bias);
    ASSERT_EQ(X_bias->rows, 3);
    ASSERT_EQ(X_bias->cols, 3);  /* original 2 + 1 bias */

    matrix_free(X_bias);
    matrix_free(X);
}

int main(void) {
    printf("Preprocessing Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_train_test_split);
    RUN_TEST(test_standardize);
    RUN_TEST(test_minmax_scale);
    RUN_TEST(test_scaler_free);
    RUN_TEST(test_add_bias_column);

    TEST_SUMMARY();
}
