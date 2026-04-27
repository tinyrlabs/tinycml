/**
 * @file test_knn.c
 * @brief Unit tests for k-Nearest Neighbors classifier
 */

#include "test_harness.h"
#include "matrix.h"
#include "knn.h"

TEST(test_knn_fit_basic) {
    /* Simple 2D data: two classes separated along x-axis */
    Matrix *X = matrix_alloc(6, 2);
    Matrix *y = matrix_alloc(6, 1);

    /* Class 0 points (left) */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0);
    /* Class 1 points (right) */
    matrix_set(X, 3, 0, 10.0); matrix_set(X, 3, 1, 0.0);
    matrix_set(X, 4, 0, 11.0); matrix_set(X, 4, 1, 0.0);
    matrix_set(X, 5, 0, 10.0); matrix_set(X, 5, 1, 1.0);

    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 0.0);
    matrix_set(y, 2, 0, 0.0);
    matrix_set(y, 3, 0, 1.0);
    matrix_set(y, 4, 0, 1.0);
    matrix_set(y, 5, 0, 1.0);

    KNNModel *model = knn_fit(X, y, 3);
    ASSERT_NOT_NULL(model);
    ASSERT_EQ(model->k, 3);
    ASSERT_NOT_NULL(model->X_train);
    ASSERT_NOT_NULL(model->y_train);

    knn_free(model);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_knn_predict_binary) {
    /* Well-separated binary classes */
    Matrix *X_train = matrix_alloc(4, 2);
    Matrix *y_train = matrix_alloc(4, 1);

    /* Class 0 */
    matrix_set(X_train, 0, 0, 0.0); matrix_set(X_train, 0, 1, 0.0);
    matrix_set(X_train, 1, 0, 0.5); matrix_set(X_train, 1, 1, 0.5);
    /* Class 1 */
    matrix_set(X_train, 2, 0, 10.0); matrix_set(X_train, 2, 1, 10.0);
    matrix_set(X_train, 3, 0, 10.5); matrix_set(X_train, 3, 1, 10.5);

    matrix_set(y_train, 0, 0, 0.0);
    matrix_set(y_train, 1, 0, 0.0);
    matrix_set(y_train, 2, 0, 1.0);
    matrix_set(y_train, 3, 0, 1.0);

    KNNModel *model = knn_fit(X_train, y_train, 1);
    ASSERT_NOT_NULL(model);

    /* Test points near each class */
    Matrix *X_test = matrix_alloc(2, 2);
    matrix_set(X_test, 0, 0, 0.1); matrix_set(X_test, 0, 1, 0.1);  /* near class 0 */
    matrix_set(X_test, 1, 0, 9.9); matrix_set(X_test, 1, 1, 9.9);  /* near class 1 */

    Matrix *pred = knn_predict(model, X_test);
    ASSERT_NOT_NULL(pred);

    ASSERT_EQ((int)matrix_get(pred, 0, 0), 0);
    ASSERT_EQ((int)matrix_get(pred, 1, 0), 1);

    matrix_free(pred);
    matrix_free(X_test);
    knn_free(model);
    matrix_free(X_train);
    matrix_free(y_train);
}

TEST(test_knn_predict_multiclass) {
    /* Three classes with 3 points each */
    Matrix *X_train = matrix_alloc(6, 1);
    Matrix *y_train = matrix_alloc(6, 1);

    /* Class 0: values around 0 */
    matrix_set(X_train, 0, 0, 0.0);
    matrix_set(X_train, 1, 0, 1.0);
    /* Class 2: values around 10 */
    matrix_set(X_train, 2, 0, 10.0);
    matrix_set(X_train, 3, 0, 11.0);
    /* Extra points for class 2 */
    matrix_set(X_train, 4, 0, 9.5);
    matrix_set(X_train, 5, 0, 10.5);

    matrix_set(y_train, 0, 0, 0.0);
    matrix_set(y_train, 1, 0, 0.0);
    matrix_set(y_train, 2, 0, 2.0);
    matrix_set(y_train, 3, 0, 2.0);
    matrix_set(y_train, 4, 0, 2.0);
    matrix_set(y_train, 5, 0, 2.0);

    KNNModel *model = knn_fit(X_train, y_train, 3);
    ASSERT_NOT_NULL(model);

    Matrix *X_test = matrix_alloc(2, 1);
    matrix_set(X_test, 0, 0, 0.5);   /* near class 0 */
    matrix_set(X_test, 1, 0, 10.2);  /* near class 2 */

    Matrix *pred = knn_predict(model, X_test);
    ASSERT_NOT_NULL(pred);

    ASSERT_EQ((int)matrix_get(pred, 0, 0), 0);
    ASSERT_EQ((int)matrix_get(pred, 1, 0), 2);

    matrix_free(pred);
    matrix_free(X_test);
    knn_free(model);
    matrix_free(X_train);
    matrix_free(y_train);
}

TEST(test_knn_null_inputs) {
    /* NULL inputs should return NULL */
    KNNModel *model = knn_fit(NULL, NULL, 3);
    ASSERT_NULL(model);

    Matrix *pred = knn_predict(NULL, NULL);
    ASSERT_NULL(pred);
}

TEST(test_knn_k1) {
    /* k=1 should perfectly memorize training data */
    Matrix *X = matrix_alloc(3, 2);
    Matrix *y = matrix_alloc(3, 1);

    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 5.0); matrix_set(X, 1, 1, 5.0);
    matrix_set(X, 2, 0, 10.0); matrix_set(X, 2, 1, 10.0);

    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 1.0);
    matrix_set(y, 2, 0, 0.0);

    KNNModel *model = knn_fit(X, y, 1);
    ASSERT_NOT_NULL(model);

    /* Predict on training data itself => should be exact */
    Matrix *pred = knn_predict(model, X);
    ASSERT_NOT_NULL(pred);

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ((int)matrix_get(pred, i, 0), (int)matrix_get(y, i, 0));
    }

    matrix_free(pred);
    knn_free(model);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("k-Nearest Neighbors Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_knn_fit_basic);
    RUN_TEST(test_knn_predict_binary);
    RUN_TEST(test_knn_predict_multiclass);
    RUN_TEST(test_knn_null_inputs);
    RUN_TEST(test_knn_k1);

    TEST_SUMMARY();
}
