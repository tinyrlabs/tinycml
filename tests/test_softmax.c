/**
 * @file test_softmax.c
 * @brief Unit tests for Softmax (multi-class) Logistic Regression
 */

#include "test_harness.h"
#include "matrix.h"
#include "logistic_regression.h"
#include "estimator.h"

TEST(test_softmax_create_free) {
    SoftmaxRegressionModel *model = softmax_model_create();
    ASSERT_NOT_NULL(model);

    ASSERT_EQ(model->base.type, MODEL_LOGISTIC_REGRESSION);
    ASSERT_EQ(model->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(model->base.is_fitted, 0);
    ASSERT(model->learning_rate > 0.0);
    ASSERT(model->max_iter > 0);
    ASSERT(model->tol > 0.0);
    ASSERT_NULL(model->weights);
    ASSERT_NULL(model->biases);
    ASSERT_EQ(model->n_features, 0);
    ASSERT_EQ(model->n_classes, 0);

    /* Free through Estimator vtable */
    model->base.free((Estimator *)model);
}

TEST(test_softmax_3class) {
    /* 3-class separable data in 2D:
     * Class 0: near (1,1)
     * Class 1: near (5,5)
     * Class 2: near (9,1)
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

    /* Class 1: near (5,5) */
    double c1[][2] = {{4.8,5.2},{5.1,4.9},{5.0,5.0},{4.9,5.1},{5.2,4.8},
                      {5.1,5.1},{4.9,4.9},{5.0,5.2},{4.8,4.8},{5.2,5.0}};
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10+i, 0, c1[i][0]);
        matrix_set(X, 10+i, 1, c1[i][1]);
        matrix_set(y, 10+i, 0, 1.0);
    }

    /* Class 2: near (9,1) */
    double c2[][2] = {{8.8,1.2},{9.1,0.9},{9.0,1.0},{8.9,1.1},{9.2,0.8},
                      {9.1,1.1},{8.9,0.9},{9.0,1.2},{8.8,0.8},{9.2,1.0}};
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 20+i, 0, c2[i][0]);
        matrix_set(X, 20+i, 1, c2[i][1]);
        matrix_set(y, 20+i, 0, 2.0);
    }

    SoftmaxRegressionModel *model = softmax_model_create();
    ASSERT_NOT_NULL(model);

    Estimator *fitted = model->base.fit((Estimator *)model, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(model->base.is_fitted, 1);
    ASSERT_EQ(model->n_classes, 3);
    ASSERT_EQ(model->n_features, 2);

    double acc = model->base.score((const Estimator *)model, X, y);
    ASSERT(acc > 0.85);

    model->base.free((Estimator *)model);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_softmax_proba) {
    /* Probabilities should be in [0,1] and rows should sum to ~1.0 */
    int N = 15;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* 3 classes with 5 samples each */
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, 1.0);
        matrix_set(X, i, 1, 1.0 + i * 0.1);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5+i, 0, 5.0);
        matrix_set(X, 5+i, 1, 5.0 + i * 0.1);
        matrix_set(y, 5+i, 0, 1.0);
    }
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 10+i, 0, 9.0);
        matrix_set(X, 10+i, 1, 1.0 + i * 0.1);
        matrix_set(y, 10+i, 0, 2.0);
    }

    SoftmaxRegressionModel *model = softmax_model_create();
    ASSERT_NOT_NULL(model);

    model->base.fit((Estimator *)model, X, y);
    ASSERT_EQ(model->base.is_fitted, 1);

    Matrix *proba = model->base.predict_proba((const Estimator *)model, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, (size_t)3);

    /* All entries in [0,1] and rows sum to ~1.0 */
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int c = 0; c < 3; c++) {
            double p = matrix_get(proba, i, c);
            ASSERT(p >= 0.0 && p <= 1.0);
            row_sum += p;
        }
        ASSERT_NEAR(row_sum, 1.0, 1e-9);
    }

    matrix_free(proba);
    model->base.free((Estimator *)model);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_softmax_score) {
    /* Verify score matches manual accuracy computation */
    int N = 15;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, 0.5);
        matrix_set(X, i, 1, 0.5 + i * 0.1);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5+i, 0, 5.5);
        matrix_set(X, 5+i, 1, 5.5 + i * 0.1);
        matrix_set(y, 5+i, 0, 1.0);
    }
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 10+i, 0, 10.5);
        matrix_set(X, 10+i, 1, 0.5 + i * 0.1);
        matrix_set(y, 10+i, 0, 2.0);
    }

    SoftmaxRegressionModel *model = softmax_model_create();
    ASSERT_NOT_NULL(model);

    model->base.fit((Estimator *)model, X, y);

    /* Get score via vtable */
    double score = model->base.score((const Estimator *)model, X, y);

    /* Compute manual accuracy */
    Matrix *pred = model->base.predict((const Estimator *)model, X);
    ASSERT_NOT_NULL(pred);
    int correct = 0;
    for (int i = 0; i < N; i++) {
        int p = (int)pred->data[i];
        int t = (int)y->data[i];
        if (p == t) correct++;
    }
    double manual_acc = (double)correct / (double)N;

    ASSERT_NEAR(score, manual_acc, 1e-12);

    matrix_free(pred);
    model->base.free((Estimator *)model);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Softmax Regression Tests\n");
    printf("========================\n\n");

    RUN_TEST(test_softmax_create_free);
    RUN_TEST(test_softmax_3class);
    RUN_TEST(test_softmax_proba);
    RUN_TEST(test_softmax_score);

    TEST_SUMMARY();
}
