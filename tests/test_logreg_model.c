/**
 * @file test_logreg_model.c
 * @brief Unit tests for LogisticRegressionModel (Estimator interface)
 */

#include "test_harness.h"
#include "matrix.h"
#include "logistic_regression.h"

TEST(test_logreg_model_create_free) {
    LogisticRegressionModel *model = logreg_model_create();
    ASSERT_NOT_NULL(model);

    ASSERT_EQ(model->base.type, MODEL_LOGISTIC_REGRESSION);
    ASSERT_EQ(model->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(model->base.is_fitted, 0);
    ASSERT(model->learning_rate > 0.0);
    ASSERT(model->max_iter > 0);
    ASSERT_NULL(model->weights);
    ASSERT_EQ(model->n_features, 0);
    ASSERT_EQ(model->n_classes, 2);

    /* Free through Estimator vtable */
    model->base.free((Estimator *)model);
}

TEST(test_logreg_model_fit_predict) {
    /* Linearly separable binary data with bias column:
     * Class 0: far left, Class 1: far right
     */
    int N = 10;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0: negative x values */
    double c0[][2] = {{1.0,-10.0},{1.0,-8.0},{1.0,-6.0},{1.0,-4.0},{1.0,-2.0}};
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, c0[i][0]);
        matrix_set(X, i, 1, c0[i][1]);
        matrix_set(y, i, 0, 0.0);
    }

    /* Class 1: positive x values */
    double c1[][2] = {{1.0,2.0},{1.0,4.0},{1.0,6.0},{1.0,8.0},{1.0,10.0}};
    for (int i = 0; i < 5; i++) {
        matrix_set(X, 5 + i, 0, c1[i][0]);
        matrix_set(X, 5 + i, 1, c1[i][1]);
        matrix_set(y, 5 + i, 0, 1.0);
    }

    LogisticRegressionModel *model = logreg_model_create();
    ASSERT_NOT_NULL(model);

    /* Fit through Estimator vtable */
    Estimator *fitted = model->base.fit((Estimator *)model, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(model->base.is_fitted, 1);
    ASSERT_EQ(model->n_features, 2);
    ASSERT_NOT_NULL(model->weights);

    /* Predict through Estimator vtable */
    Matrix *pred = model->base.predict((const Estimator *)model, X);
    ASSERT_NOT_NULL(pred);

    /* Accuracy should be > 0.8 */
    double acc = model->base.score((const Estimator *)model, X, y);
    ASSERT(acc > 0.8);

    matrix_free(pred);
    model->base.free((Estimator *)model);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_logreg_model_score) {
    /* Verify score matches manual accuracy computation */
    int N = 8;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Simple separable data */
    double data[][2] = {
        {1.0, -7.0}, {1.0, -5.0}, {1.0, -3.0}, {1.0, -1.0},
        {1.0,  1.0}, {1.0,  3.0}, {1.0,  5.0}, {1.0,  7.0}
    };
    for (int i = 0; i < N; i++) {
        matrix_set(X, i, 0, data[i][0]);
        matrix_set(X, i, 1, data[i][1]);
        matrix_set(y, i, 0, i < 4 ? 0.0 : 1.0);
    }

    LogisticRegressionModel *model = logreg_model_create();
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

TEST(test_logreg_model_proba) {
    /* Probabilities should be in [0,1] and each row sums to ~1.0 */
    int N = 6;
    Matrix *X = matrix_alloc(N, 2);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    double data[][2] = {
        {1.0, -10.0}, {1.0, -5.0}, {1.0, -1.0},
        {1.0,  1.0},  {1.0,  5.0}, {1.0, 10.0}
    };
    for (int i = 0; i < N; i++) {
        matrix_set(X, i, 0, data[i][0]);
        matrix_set(X, i, 1, data[i][1]);
        matrix_set(y, i, 0, i < 3 ? 0.0 : 1.0);
    }

    LogisticRegressionModel *model = logreg_model_create();
    ASSERT_NOT_NULL(model);

    model->base.fit((Estimator *)model, X, y);
    ASSERT_EQ(model->base.is_fitted, 1);

    /* Get probability matrix via vtable */
    Matrix *proba = model->base.predict_proba((const Estimator *)model, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, (size_t)2);

    /* All entries in [0,1] and rows sum to ~1.0 */
    for (int i = 0; i < N; i++) {
        double p0 = matrix_get(proba, i, 0);
        double p1 = matrix_get(proba, i, 1);
        ASSERT(p0 >= 0.0 && p0 <= 1.0);
        ASSERT(p1 >= 0.0 && p1 <= 1.0);
        ASSERT_NEAR(p0 + p1, 1.0, 1e-9);
    }

    matrix_free(proba);
    model->base.free((Estimator *)model);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Logistic Regression Model (Estimator) Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_logreg_model_create_free);
    RUN_TEST(test_logreg_model_fit_predict);
    RUN_TEST(test_logreg_model_score);
    RUN_TEST(test_logreg_model_proba);

    TEST_SUMMARY();
}
