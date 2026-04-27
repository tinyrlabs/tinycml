/**
 * @file knn.c
 * @brief Implementation of k-Nearest Neighbors classifier
 */

#include "knn.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* Structure for sorting neighbors by distance */
typedef struct {
    double distance;
    double label;
} Neighbor;

/* Compare function for qsort */
static int compare_neighbors(const void *a, const void *b) {
    double da = ((const Neighbor *)a)->distance;
    double db = ((const Neighbor *)b)->distance;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Compute Euclidean distance between two rows */
static double euclidean_distance(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k) {
    if (!X || !y || X->rows != y->rows || k <= 0) {
        return NULL;
    }

    KNNModel *model = malloc(sizeof(KNNModel));
    if (!model) {
        return NULL;
    }

    model->X_train = matrix_copy(X);
    model->y_train = matrix_copy(y);
    model->k = k;

    if (!model->X_train || !model->y_train) {
        knn_free(model);
        return NULL;
    }

    return model;
}

Matrix* knn_predict(const KNNModel *model, const Matrix *X) {
    if (!model || !X || X->cols != model->X_train->cols) {
        return NULL;
    }

    size_t n_train = model->X_train->rows;
    size_t n_test = X->rows;
    size_t n_features = X->cols;
    int k = model->k;

    /* Limit k to training set size */
    if ((size_t)k > n_train) {
        k = (int)n_train;
    }

    Matrix *predictions = matrix_alloc(n_test, 1);
    if (!predictions) {
        return NULL;
    }

    Neighbor *neighbors = malloc(n_train * sizeof(Neighbor));
    if (!neighbors) {
        matrix_free(predictions);
        return NULL;
    }

    /* For each test sample */
    for (size_t i = 0; i < n_test; i++) {
        /* Compute distances to all training samples */
        for (size_t j = 0; j < n_train; j++) {
            neighbors[j].distance = euclidean_distance(
                &X->data[i * n_features],
                &model->X_train->data[j * n_features],
                n_features
            );
            neighbors[j].label = model->y_train->data[j];
        }

        /* Sort by distance */
        qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

        /* Majority vote among k nearest neighbors */
        /* Simple approach: count votes for each class */
        /* Assuming binary classification or small number of classes */
        double max_label = 0;
        for (size_t j = 0; j < (size_t)k; j++) {
            if (neighbors[j].label > max_label) {
                max_label = neighbors[j].label;
            }
        }

        /* Count votes for each class (0 to max_label) */
        int n_classes = (int)max_label + 1;
        int *votes = calloc(n_classes, sizeof(int));
        if (!votes) {
            free(neighbors);
            matrix_free(predictions);
            return NULL;
        }

        for (int j = 0; j < k; j++) {
            int class_idx = (int)neighbors[j].label;
            if (class_idx >= 0 && class_idx < n_classes) {
                votes[class_idx]++;
            }
        }

        /* Find majority class */
        int max_votes = 0;
        int predicted_class = 0;
        for (int c = 0; c < n_classes; c++) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                predicted_class = c;
            }
        }

        predictions->data[i] = (double)predicted_class;
        free(votes);
    }

    free(neighbors);
    return predictions;
}

Matrix* knn_predict_proba(const KNNModel *model, const Matrix *X, int n_classes) {
    if (!model || !X || X->cols != model->X_train->cols || n_classes <= 0) {
        return NULL;
    }

    size_t n_train = model->X_train->rows;
    size_t n_test = X->rows;
    size_t n_features = X->cols;
    int k = model->k;

    /* Limit k to training set size */
    if ((size_t)k > n_train) {
        k = (int)n_train;
    }

    Matrix *proba = matrix_alloc(n_test, (size_t)n_classes);
    if (!proba) return NULL;

    Neighbor *neighbors = malloc(n_train * sizeof(Neighbor));
    if (!neighbors) {
        matrix_free(proba);
        return NULL;
    }

    /* For each test sample */
    for (size_t i = 0; i < n_test; i++) {
        /* Compute distances to all training samples */
        for (size_t j = 0; j < n_train; j++) {
            neighbors[j].distance = euclidean_distance(
                &X->data[i * n_features],
                &model->X_train->data[j * n_features],
                n_features
            );
            neighbors[j].label = model->y_train->data[j];
        }

        /* Sort by distance */
        qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

        /* Count class frequencies among k nearest neighbors */
        int *votes = calloc((size_t)n_classes, sizeof(int));
        if (!votes) {
            free(neighbors);
            matrix_free(proba);
            return NULL;
        }

        for (int j = 0; j < k; j++) {
            int class_idx = (int)neighbors[j].label;
            if (class_idx >= 0 && class_idx < n_classes) {
                votes[class_idx]++;
            }
        }

        /* Probability = class_count / k */
        for (int c = 0; c < n_classes; c++) {
            proba->data[i * (size_t)n_classes + c] = (double)votes[c] / (double)k;
        }

        free(votes);
    }

    free(neighbors);
    return proba;
}

void knn_free(KNNModel *model) {
    if (model) {
        matrix_free(model->X_train);
        matrix_free(model->y_train);
        free(model);
    }
}
