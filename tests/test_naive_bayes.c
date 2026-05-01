/**
 * @file test_naive_bayes.c
 * @brief Unit tests for Gaussian Naive Bayes classifier
 */

#include "test_harness.h"
#include "matrix.h"
#include "naive_bayes.h"
#include "metrics.h"

TEST(test_nb_create_free) {
    GaussianNaiveBayes *nb = gaussian_nb_create();
    ASSERT_NOT_NULL(nb);
    ASSERT_EQ(nb->base.type, MODEL_NAIVE_BAYES);
    ASSERT_EQ(nb->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(nb->base.is_fitted, 0);
    ASSERT(nb->var_smoothing > 0.0);

    nb->base.free((Estimator *)nb);
}

TEST(test_nb_iris_like) {
    /* 2D classification with 3 well-separated Gaussian clusters.
     * Class 0: centered around (1,1)
     * Class 1: centered around (4,4)
     * Class 2: centered around (7,1)
     * 10 samples per class = 30 total.
     */
    int N = 30;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0: near (1,1) */
    double c0[][2] = {{0.8,1.2},{1.1,0.9},{1.0,1.0},{0.9,1.1},{1.2,0.8},
                      {1.1,1.1},{0.9,0.9},{1.0,1.2},{0.8,0.8},{1.2,1.0}};
    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, c0[i][0]);
        matrix_set(X, i, 1, c0[i][1]);
        matrix_set(y, i, 0, 0.0);
    }

    /* Class 1: near (4,4) */
    double c1[][2] = {{3.8,4.2},{4.1,3.9},{4.0,4.0},{3.9,4.1},{4.2,3.8},
                      {4.1,4.1},{3.9,3.9},{4.0,4.2},{3.8,3.8},{4.2,4.0}};
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10+i, 0, c1[i][0]);
        matrix_set(X, 10+i, 1, c1[i][1]);
        matrix_set(y, 10+i, 0, 1.0);
    }

    /* Class 2: near (7,1) */
    double c2[][2] = {{6.8,1.2},{7.1,0.9},{7.0,1.0},{6.9,1.1},{7.2,0.8},
                      {7.1,1.1},{6.9,0.9},{7.0,1.2},{6.8,0.8},{7.2,1.0}};
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 20+i, 0, c2[i][0]);
        matrix_set(X, 20+i, 1, c2[i][1]);
        matrix_set(y, 20+i, 0, 2.0);
    }

    GaussianNaiveBayes *nb = gaussian_nb_create();
    ASSERT_NOT_NULL(nb);

    Estimator *fitted = nb->base.fit((Estimator *)nb, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(nb->base.is_fitted, 1);
    ASSERT_EQ(nb->n_classes, 3);
    ASSERT_EQ(nb->n_features, 2);

    /* Predict on training data — should be perfect or near-perfect */
    Matrix *pred = nb->base.predict((const Estimator *)nb, X);
    ASSERT_NOT_NULL(pred);

    double acc = nb->base.score((const Estimator *)nb, X, y);
    ASSERT(acc > 0.8);

    matrix_free(pred);
    nb->base.free((Estimator *)nb);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_nb_predict_proba) {
    /* Simple binary: class 0 near (0,0), class 1 near (5,5) */
    int N = 10;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0 */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, 0.0 + i * 0.1);
        matrix_set(X, i, 1, 0.0 + i * 0.1);
        matrix_set(y, i, 0, 0.0);
    }
    /* Class 1 */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5+i, 0, 5.0 + i * 0.1);
        matrix_set(X, 5+i, 1, 5.0 + i * 0.1);
        matrix_set(y, 5+i, 0, 1.0);
    }

    GaussianNaiveBayes *nb = gaussian_nb_create();
    ASSERT_NOT_NULL(nb);

    nb->base.fit((Estimator *)nb, X, y);
    ASSERT_EQ(nb->base.is_fitted, 1);

    Matrix *proba = gaussian_nb_predict_proba((const Estimator *)nb, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, (size_t)2);

    /* Each row should sum to ~1.0 and all entries in [0,1] */
    for (int i = 0; i < N; i++) {
        double p0 = matrix_get(proba, i, 0);
        double p1 = matrix_get(proba, i, 1);
        ASSERT(p0 >= 0.0 && p0 <= 1.0);
        ASSERT(p1 >= 0.0 && p1 <= 1.0);
        ASSERT_NEAR(p0 + p1, 1.0, 1e-9);
    }

    /* Samples near (0,0) should have high P(class 0) */
    ASSERT(matrix_get(proba, 0, 0) > 0.5);
    /* Samples near (5,5) should have high P(class 1) */
    ASSERT(matrix_get(proba, 5, 1) > 0.5);

    matrix_free(proba);
    nb->base.free((Estimator *)nb);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_nb_binary) {
    /* Binary classification: two separable groups along x-axis */
    int N = 20;
    Matrix *X = matrix_alloc(N, 1);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0: values around -3 */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, -3.0 + (i % 5) * 0.2);
        matrix_set(y, i, 0, 0.0);
    }
    /* Class 1: values around +3 */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10+i, 0, 3.0 + (i % 5) * 0.2);
        matrix_set(y, 10+i, 0, 1.0);
    }

    GaussianNaiveBayes *nb = gaussian_nb_create();
    ASSERT_NOT_NULL(nb);

    nb->base.fit((Estimator *)nb, X, y);
    ASSERT_EQ(nb->base.is_fitted, 1);
    ASSERT_EQ(nb->n_classes, 2);

    /* Predict on training data — should be perfect */
    double acc = nb->base.score((const Estimator *)nb, X, y);
    ASSERT(acc > 0.9);

    /* Test a new sample */
    Matrix *X_test = matrix_alloc(2, 1);
    matrix_set(X_test, 0, 0, -3.0);  /* Should be class 0 */
    matrix_set(X_test, 1, 0,  3.0);  /* Should be class 1 */

    Matrix *pred = nb->base.predict((const Estimator *)nb, X_test);
    ASSERT_NOT_NULL(pred);
    ASSERT_EQ((int)pred->data[0], 0);
    ASSERT_EQ((int)pred->data[1], 1);

    matrix_free(pred);
    matrix_free(X_test);
    nb->base.free((Estimator *)nb);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    RUN_TEST(test_nb_create_free);
    RUN_TEST(test_nb_iris_like);
    RUN_TEST(test_nb_predict_proba);
    RUN_TEST(test_nb_binary);
    TEST_SUMMARY();
}
