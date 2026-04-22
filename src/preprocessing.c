/**
 * @file preprocessing.c
 * @brief Implementation of data preprocessing functions
 */

#include "preprocessing.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed) {
    TrainTestSplit split = {NULL, NULL, NULL, NULL};

    if (!X || !y || X->rows != y->rows) {
        return split;
    }

    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        return split;
    }

    size_t n = X->rows;
    size_t n_test = (size_t)(n * test_ratio);
    size_t n_train = n - n_test;

    if (n_train == 0 || n_test == 0) {
        return split;
    }

    /* Create shuffled indices */
    size_t *indices = malloc(n * sizeof(size_t));
    if (!indices) {
        return split;
    }

    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }

    rand_seed(seed);
    shuffle_indices(indices, n);

    /* Allocate matrices */
    split.X_train = matrix_alloc(n_train, X->cols);
    split.X_test = matrix_alloc(n_test, X->cols);
    split.y_train = matrix_alloc(n_train, y->cols);
    split.y_test = matrix_alloc(n_test, y->cols);

    if (!split.X_train || !split.X_test || !split.y_train || !split.y_test) {
        train_test_split_free(&split);
        free(indices);
        return (TrainTestSplit){NULL, NULL, NULL, NULL};
    }

    /* Copy training data */
    for (size_t i = 0; i < n_train; i++) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X->cols; j++) {
            split.X_train->data[i * X->cols + j] = X->data[idx * X->cols + j];
        }
        for (size_t j = 0; j < y->cols; j++) {
            split.y_train->data[i * y->cols + j] = y->data[idx * y->cols + j];
        }
    }

    /* Copy test data */
    for (size_t i = 0; i < n_test; i++) {
        size_t idx = indices[n_train + i];
        for (size_t j = 0; j < X->cols; j++) {
            split.X_test->data[i * X->cols + j] = X->data[idx * X->cols + j];
        }
        for (size_t j = 0; j < y->cols; j++) {
            split.y_test->data[i * y->cols + j] = y->data[idx * y->cols + j];
        }
    }

    free(indices);
    return split;
}

void train_test_split_free(TrainTestSplit *split) {
    if (split) {
        matrix_free(split->X_train);
        matrix_free(split->X_test);
        matrix_free(split->y_train);
        matrix_free(split->y_test);
        split->X_train = NULL;
        split->X_test = NULL;
        split->y_train = NULL;
        split->y_test = NULL;
    }
}

Scaler* standardize_fit(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    Scaler *scaler = malloc(sizeof(Scaler));
    if (!scaler) {
        return NULL;
    }

    scaler->n_features = X->cols;
    scaler->means = malloc(X->cols * sizeof(double));
    scaler->stds = malloc(X->cols * sizeof(double));

    if (!scaler->means || !scaler->stds) {
        scaler_free(scaler);
        return NULL;
    }

    /* Compute mean and std for each column */
    for (size_t j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            sum += X->data[i * X->cols + j];
        }
        scaler->means[j] = sum / X->rows;

        double var_sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            double diff = X->data[i * X->cols + j] - scaler->means[j];
            var_sum += diff * diff;
        }
        scaler->stds[j] = sqrt(var_sum / X->rows);

        /* Avoid division by zero */
        if (scaler->stds[j] < DBL_EPSILON) {
            scaler->stds[j] = 1.0;
        }
    }

    return scaler;
}

Matrix* standardize_transform(const Matrix *X, const Scaler *scaler) {
    if (!X || !scaler || X->cols != scaler->n_features) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        for (size_t j = 0; j < X->cols; j++) {
            result->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - scaler->means[j]) / scaler->stds[j];
        }
    }

    return result;
}

Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out) {
    Scaler *scaler = standardize_fit(X);
    if (!scaler) {
        return NULL;
    }

    Matrix *result = standardize_transform(X, scaler);
    if (!result) {
        scaler_free(scaler);
        return NULL;
    }

    if (scaler_out) {
        *scaler_out = scaler;
    } else {
        scaler_free(scaler);
    }

    return result;
}

void scaler_free(Scaler *scaler) {
    if (scaler) {
        free(scaler->means);
        free(scaler->stds);
        free(scaler);
    }
}

MinMaxScaler* minmax_fit(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    MinMaxScaler *scaler = malloc(sizeof(MinMaxScaler));
    if (!scaler) {
        return NULL;
    }

    scaler->n_features = X->cols;
    scaler->mins = malloc(X->cols * sizeof(double));
    scaler->maxs = malloc(X->cols * sizeof(double));

    if (!scaler->mins || !scaler->maxs) {
        minmax_scaler_free(scaler);
        return NULL;
    }

    /* Find min and max for each column */
    for (size_t j = 0; j < X->cols; j++) {
        scaler->mins[j] = DBL_MAX;
        scaler->maxs[j] = -DBL_MAX;

        for (size_t i = 0; i < X->rows; i++) {
            double val = X->data[i * X->cols + j];
            if (val < scaler->mins[j]) {
                scaler->mins[j] = val;
            }
            if (val > scaler->maxs[j]) {
                scaler->maxs[j] = val;
            }
        }

        /* Avoid division by zero */
        if (scaler->maxs[j] - scaler->mins[j] < DBL_EPSILON) {
            scaler->maxs[j] = scaler->mins[j] + 1.0;
        }
    }

    return scaler;
}

Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler) {
    if (!X || !scaler || X->cols != scaler->n_features) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        for (size_t j = 0; j < X->cols; j++) {
            double range = scaler->maxs[j] - scaler->mins[j];
            result->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - scaler->mins[j]) / range;
        }
    }

    return result;
}

void minmax_scaler_free(MinMaxScaler *scaler) {
    if (scaler) {
        free(scaler->mins);
        free(scaler->maxs);
        free(scaler);
    }
}

Matrix* add_bias_column(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols + 1);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        result->data[i * result->cols] = 1.0;  /* Bias column */
        for (size_t j = 0; j < X->cols; j++) {
            result->data[i * result->cols + j + 1] = X->data[i * X->cols + j];
        }
    }

    return result;
}
