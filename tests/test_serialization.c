/**
 * @file test_serialization.c
 * @brief Tests for model save/load (serialization) across all supported models
 */

#include "estimator.h"
#include "logistic_regression.h"
#include "naive_bayes.h"
#include "svm.h"
#include "linear_regression.h"
#include "matrix.h"
#include "test_harness.h"
#include <unistd.h>

/* Helper: create simple binary classification data */
static Matrix* make_binary_data(int n) {
    Matrix *X = matrix_alloc(n, 2);
    for (int i = 0; i < n; i++) {
        double x1 = (double)(i % 10) / 10.0;
        double x2 = (double)((i+3) % 8) / 8.0;
        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);
    }
    return X;
}

static Matrix* make_binary_labels(int n) {
    Matrix *y = matrix_alloc(n, 1);
    for (int i = 0; i < n; i++) {
        matrix_set(y, i, 0, (i % 2 == 0) ? 0.0 : 1.0);
    }
    return y;
}

TEST(test_save_load_logreg) {
    Matrix *X = make_binary_data(50);
    Matrix *y = make_binary_labels(50);

    LogisticRegressionModel *model = logreg_model_create();
    model->base.fit(&model->base, X, y);

    Matrix *pred_before = model->base.predict(&model->base, X);

    int save_ok = model->base.save(&model->base, "/tmp/test_logreg.cml");
    ASSERT(save_ok == 0);

    Estimator *loaded_est = logreg_model_load("/tmp/test_logreg.cml");
    ASSERT(loaded_est != NULL);

    Matrix *pred_after = loaded_est->predict(loaded_est, X);
    /* Predictions should match */
    int match = 1;
    for (size_t i = 0; i < pred_before->rows; i++) {
        if (matrix_get(pred_before, i, 0) != matrix_get(pred_after, i, 0)) {
            match = 0; break;
        }
    }
    ASSERT(match);

    matrix_free(pred_after);
    loaded_est->free(loaded_est);
    matrix_free(pred_before);
    matrix_free(X); matrix_free(y);
    model->base.free(&model->base);
    unlink("/tmp/test_logreg.cml");
}

TEST(test_save_load_gaussian_nb) {
    Matrix *X = make_binary_data(50);
    Matrix *y = make_binary_labels(50);

    GaussianNaiveBayes *model = gaussian_nb_create();
    model->base.fit(&model->base, X, y);

    Matrix *pred_before = model->base.predict(&model->base, X);

    int save_ok = model->base.save(&model->base, "/tmp/test_nb.cml");
    ASSERT(save_ok == 0);

    Estimator *loaded_est = gaussian_nb_load("/tmp/test_nb.cml");
    ASSERT(loaded_est != NULL);

    Matrix *pred_after = loaded_est->predict(loaded_est, X);
    int match = 1;
    for (size_t i = 0; i < pred_before->rows && i < pred_after->rows; i++) {
        if (matrix_get(pred_before, i, 0) != matrix_get(pred_after, i, 0)) {
            match = 0; break;
        }
    }
    ASSERT(match);

    matrix_free(pred_after);
    loaded_est->free(loaded_est);
    matrix_free(pred_before);
    matrix_free(X); matrix_free(y);
    model->base.free(&model->base);
    unlink("/tmp/test_nb.cml");
}

TEST(test_save_load_svm) {
    Matrix *X = make_binary_data(50);
    Matrix *y = make_binary_labels(50);

    /* Convert labels to {-1, +1} for SVM */
    Matrix *y_svm = matrix_alloc(50, 1);
    for (int i = 0; i < 50; i++) {
        matrix_set(y_svm, i, 0, matrix_get(y, i, 0) == 1.0 ? 1.0 : -1.0);
    }

    LinearSVC *model = linear_svc_create();
    model->base.fit(&model->base, X, y_svm);

    int save_ok = model->base.save(&model->base, "/tmp/test_svm.cml");
    ASSERT(save_ok == 0);

    Estimator *loaded_est = linear_svc_load("/tmp/test_svm.cml");
    ASSERT(loaded_est != NULL);

    Matrix *pred_before = model->base.predict(&model->base, X);
    Matrix *pred_after = loaded_est->predict(loaded_est, X);
    int match = 1;
    for (size_t i = 0; i < pred_before->rows && i < pred_after->rows; i++) {
        if (matrix_get(pred_before, i, 0) != matrix_get(pred_after, i, 0)) {
            match = 0; break;
        }
    }
    ASSERT(match);

    matrix_free(pred_before);
    matrix_free(pred_after);
    loaded_est->free(loaded_est);
    matrix_free(y_svm);
    matrix_free(X); matrix_free(y);
    model->base.free(&model->base);
    unlink("/tmp/test_svm.cml");
}

TEST(test_save_load_linreg) {
    Matrix *X = matrix_alloc(30, 2);
    Matrix *y = matrix_alloc(30, 1);
    for (int i = 0; i < 30; i++) {
        double x1 = (double)i / 30.0;
        double x2 = (double)(i * i % 30) / 30.0;
        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);
        matrix_set(y, i, 0, 3.0 * x1 + 2.0 * x2 + 0.5);
    }

    LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);
    model->base.fit(&model->base, X, y);

    Matrix *pred_before = model->base.predict(&model->base, X);

    int save_ok = model->base.save(&model->base, "/tmp/test_linreg.cml");
    ASSERT(save_ok == 0);

    Estimator *loaded_est = linear_regression_load("/tmp/test_linreg.cml");
    ASSERT(loaded_est != NULL);

    Matrix *pred_after = loaded_est->predict(loaded_est, X);
    /* For regression, check predictions are close (within 1e-6) */
    int close = 1;
    for (size_t i = 0; i < pred_before->rows && i < pred_after->rows; i++) {
        double diff = matrix_get(pred_before, i, 0) - matrix_get(pred_after, i, 0);
        if (diff < 0) diff = -diff;
        if (diff > 1e-6) { close = 0; break; }
    }
    ASSERT(close);

    matrix_free(pred_after);
    loaded_est->free(loaded_est);
    matrix_free(pred_before);
    matrix_free(X); matrix_free(y);
    model->base.free(&model->base);
    unlink("/tmp/test_linreg.cml");
}

TEST(test_save_load_multinomial_nb) {
    /* Multinomial NB needs non-negative features */
    Matrix *X = matrix_alloc(50, 3);
    Matrix *y = matrix_alloc(50, 1);
    for (int i = 0; i < 50; i++) {
        matrix_set(X, i, 0, (double)(i % 5) + 1.0);
        matrix_set(X, i, 1, (double)((i+2) % 7) + 1.0);
        matrix_set(X, i, 2, (double)((i+1) % 4) + 1.0);
        matrix_set(y, i, 0, (double)(i % 3));
    }

    MultinomialNB *model = multinomial_nb_create();
    model->base.fit(&model->base, X, y);

    Matrix *pred_before = model->base.predict(&model->base, X);

    int save_ok = model->base.save(&model->base, "/tmp/test_mnb.cml");
    ASSERT(save_ok == 0);

    Estimator *loaded_est = multinomial_nb_load("/tmp/test_mnb.cml");
    ASSERT(loaded_est != NULL);

    Matrix *pred_after = loaded_est->predict(loaded_est, X);
    int match = 1;
    for (size_t i = 0; i < pred_before->rows && i < pred_after->rows; i++) {
        if (matrix_get(pred_before, i, 0) != matrix_get(pred_after, i, 0)) {
            match = 0; break;
        }
    }
    ASSERT(match);

    matrix_free(pred_after);
    loaded_est->free(loaded_est);
    matrix_free(pred_before);
    matrix_free(X); matrix_free(y);
    model->base.free(&model->base);
    unlink("/tmp/test_mnb.cml");
}

int main(void) {
    RUN_TEST(test_save_load_logreg);
    RUN_TEST(test_save_load_gaussian_nb);
    RUN_TEST(test_save_load_svm);
    RUN_TEST(test_save_load_linreg);
    RUN_TEST(test_save_load_multinomial_nb);

    TEST_SUMMARY();
}
