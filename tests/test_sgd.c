/**
 * test_sgd.c - Tests for SGD classifier and regressor
 */

#include "test_harness.h"
#include "sgd.h"
#include "matrix.h"

TEST(test_sgd_create) {
    SGDModel *m = sgd_classifier_create(0.0001, 0.01, 100);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->loss, SGD_LOSS_LOG);
    sgd_free((Estimator*)m);
}

TEST(test_sgd_classification) {
    /* Linearly separable: class 0 for x<0, class 1 for x>=0 */
    size_t n = 100;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        double x0 = ((double)i - 50.0) / 10.0;
        double x1 = ((double)i - 50.0) / 10.0;
        matrix_set(X, i, 0, x0);
        matrix_set(X, i, 1, x1);
        matrix_set(y, i, 0, x0 >= 0 ? 1.0 : 0.0);
    }

    SGDModel *m = sgd_classifier_create(0.0001, 0.1, 200);
    Estimator *fitted = sgd_fit((Estimator*)m, X, y);
    ASSERT_NOT_NULL(fitted);

    double score = sgd_score((Estimator*)m, X, y);
    ASSERT(score > 0.85);

    /* Test predict_proba */
    Matrix *proba = sgd_predict_proba((Estimator*)m, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ((int)proba->cols, 2);
    matrix_free(proba);

    matrix_free(X);
    matrix_free(y);
    sgd_free((Estimator*)m);
}

TEST(test_sgd_regression) {
    /* y = 3*x + 2 */
    size_t n = 50;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        double x = (double)i / 5.0;
        matrix_set(X, i, 0, x);
        matrix_set(y, i, 0, 3.0 * x + 2.0);
    }

    SGDModel *m = sgd_regressor_create(0.0001, 0.01, 500);
    sgd_fit((Estimator*)m, X, y);

    double score = sgd_score((Estimator*)m, X, y);
    ASSERT(score > 0.9);

    matrix_free(X);
    matrix_free(y);
    sgd_free((Estimator*)m);
}

TEST(test_sgd_save_load) {
    size_t n = 30;
    Matrix *X = matrix_alloc(n, 1);
    Matrix *y = matrix_alloc(n, 1);

    for (size_t i = 0; i < n; i++) {
        matrix_set(X, i, 0, (double)i / 5.0);
        matrix_set(y, i, 0, (i >= 15) ? 1.0 : 0.0);
    }

    SGDModel *m = sgd_classifier_create(0.0001, 0.1, 100);
    sgd_fit((Estimator*)m, X, y);

    Matrix *pred_before = sgd_predict((Estimator*)m, X);
    ASSERT_NOT_NULL(pred_before);

    int rc = sgd_save((Estimator*)m, "/tmp/test_sgd.bin");
    ASSERT_EQ(rc, 0);

    Estimator *loaded = sgd_load("/tmp/test_sgd.bin");
    ASSERT_NOT_NULL(loaded);

    Matrix *pred_after = sgd_predict(loaded, X);
    ASSERT_NOT_NULL(pred_after);

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ((int)pred_before->data[i], (int)pred_after->data[i]);
    }

    matrix_free(pred_before);
    matrix_free(pred_after);
    matrix_free(X);
    matrix_free(y);
    sgd_free((Estimator*)m);
    sgd_free(loaded);
    remove("/tmp/test_sgd.bin");
}

int main(void) {
    printf("=== SGD Tests ===\n\n");
    RUN_TEST(test_sgd_create);
    RUN_TEST(test_sgd_classification);
    RUN_TEST(test_sgd_regression);
    RUN_TEST(test_sgd_save_load);
    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return 0;
}
