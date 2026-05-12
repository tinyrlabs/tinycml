/**
 * @file test_knn_proba.c
 * @brief Unit tests for KNN predict_proba
 */

#include "test_harness.h"
#include "matrix.h"
#include "knn.h"

TEST(test_knn_proba_range) {
    /* Train KNN with k=3 on 3-class data, verify probabilities in [0,1] and sum ~1.0 */
    int N = 9;
    int n_classes = 3;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* 3 classes, well separated in 2D */
    /* Class 0 */
    matrix_set(X, 0, 0, 0.0);  matrix_set(X, 0, 1, 0.0);  y->data[0] = 0.0;
    matrix_set(X, 1, 0, 0.1);  matrix_set(X, 1, 1, 0.0);  y->data[1] = 0.0;
    matrix_set(X, 2, 0, 0.0);  matrix_set(X, 2, 1, 0.1);  y->data[2] = 0.0;
    /* Class 1 */
    matrix_set(X, 3, 0, 5.0);  matrix_set(X, 3, 1, 5.0);  y->data[3] = 1.0;
    matrix_set(X, 4, 0, 5.1);  matrix_set(X, 4, 1, 5.0);  y->data[4] = 1.0;
    matrix_set(X, 5, 0, 5.0);  matrix_set(X, 5, 1, 5.1);  y->data[5] = 1.0;
    /* Class 2 */
    matrix_set(X, 6, 0, 10.0); matrix_set(X, 6, 1, 10.0); y->data[6] = 2.0;
    matrix_set(X, 7, 0, 10.1); matrix_set(X, 7, 1, 10.0); y->data[7] = 2.0;
    matrix_set(X, 8, 0, 10.0); matrix_set(X, 8, 1, 10.1); y->data[8] = 2.0;

    KNNModel *model = knn_fit(X, y, 3);
    ASSERT_NOT_NULL(model);

    Matrix *proba = knn_predict_proba(model, X, n_classes);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, (size_t)n_classes);

    /* Each row should sum to ~1.0 and all values in [0, 1] */
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int c = 0; c < n_classes; c++) {
            double p = proba->data[i * n_classes + c];
            ASSERT(p >= 0.0);
            ASSERT(p <= 1.0);
            row_sum += p;
        }
        ASSERT_NEAR(row_sum, 1.0, 1e-10);
    }

    matrix_free(proba);
    knn_free(model);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_knn_proba_k1) {
    /* k=1 should give probabilities of exactly 0.0 or 1.0 */
    int N = 4;
    int n_classes = 2;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0 */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); y->data[0] = 0.0;
    matrix_set(X, 1, 0, 0.5); matrix_set(X, 1, 1, 0.5); y->data[1] = 0.0;
    /* Class 1 */
    matrix_set(X, 2, 0, 10.0); matrix_set(X, 2, 1, 10.0); y->data[2] = 1.0;
    matrix_set(X, 3, 0, 10.5); matrix_set(X, 3, 1, 10.5); y->data[3] = 1.0;

    KNNModel *model = knn_fit(X, y, 1);
    ASSERT_NOT_NULL(model);

    Matrix *proba = knn_predict_proba(model, X, n_classes);
    ASSERT_NOT_NULL(proba);

    /* With k=1, each probability should be exactly 0.0 or 1.0 */
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < n_classes; c++) {
            double p = proba->data[i * n_classes + c];
            ASSERT(p == 0.0 || p == 1.0);
        }
    }

    /* Each sample's nearest neighbor is itself, so proba should reflect the true class */
    /* Sample 0 (class 0): proba = [1.0, 0.0] */
    ASSERT_NEAR(proba->data[0 * 2 + 0], 1.0, 1e-10);
    ASSERT_NEAR(proba->data[0 * 2 + 1], 0.0, 1e-10);
    /* Sample 2 (class 1): proba = [0.0, 1.0] */
    ASSERT_NEAR(proba->data[2 * 2 + 0], 0.0, 1e-10);
    ASSERT_NEAR(proba->data[2 * 2 + 1], 1.0, 1e-10);

    matrix_free(proba);
    knn_free(model);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("KNN Predict Proba Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_knn_proba_range);
    RUN_TEST(test_knn_proba_k1);

    TEST_SUMMARY();
}
