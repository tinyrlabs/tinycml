/**
 * @file kmeans.c
 * @brief Implementation of k-Means clustering
 */

#include "kmeans.h"
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Compute Euclidean distance between two vectors */
static double euclidean_distance(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* Find nearest centroid for a data point */
static int find_nearest_centroid(const double *point, const Matrix *centroids) {
    int nearest = 0;
    double min_dist = DBL_MAX;

    for (size_t c = 0; c < centroids->rows; c++) {
        double dist = euclidean_distance(point, &centroids->data[c * centroids->cols], centroids->cols);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = (int)c;
        }
    }

    return nearest;
}

KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed) {
    if (!X || k <= 0 || max_iter <= 0) {
        return NULL;
    }

    size_t n = X->rows;
    size_t d = X->cols;

    if ((size_t)k > n) {
        k = (int)n;
    }

    KMeansModel *model = malloc(sizeof(KMeansModel));
    if (!model) {
        return NULL;
    }

    model->k = k;
    model->max_iter = max_iter;
    model->centroids = matrix_alloc(k, d);

    if (!model->centroids) {
        free(model);
        return NULL;
    }

    /* Initialize centroids randomly from data points */
    rand_seed(seed);
    size_t *used = calloc(n, sizeof(size_t));
    if (!used) {
        kmeans_free(model);
        return NULL;
    }

    for (int c = 0; c < k; c++) {
        size_t idx;
        do {
            idx = (size_t)(rand_uniform() * n);
        } while (used[idx] && c > 0);
        used[idx] = 1;

        for (size_t j = 0; j < d; j++) {
            model->centroids->data[c * d + j] = X->data[idx * d + j];
        }
    }
    free(used);

    /* Cluster assignments */
    int *assignments = malloc(n * sizeof(int));
    if (!assignments) {
        kmeans_free(model);
        return NULL;
    }

    /* Lloyd's algorithm */
    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;

        /* Assignment step: assign each point to nearest centroid */
        for (size_t i = 0; i < n; i++) {
            int nearest = find_nearest_centroid(&X->data[i * d], model->centroids);
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = 1;
            }
        }

        /* Check for convergence */
        if (!changed && iter > 0) {
            break;
        }

        /* Update step: recompute centroids as mean of assigned points */
        int *counts = calloc(k, sizeof(int));
        if (!counts) {
            free(assignments);
            kmeans_free(model);
            return NULL;
        }

        /* Reset centroids */
        memset(model->centroids->data, 0, k * d * sizeof(double));

        /* Sum assigned points */
        for (size_t i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            for (size_t j = 0; j < d; j++) {
                model->centroids->data[c * d + j] += X->data[i * d + j];
            }
        }

        /* Compute means */
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < d; j++) {
                    model->centroids->data[c * d + j] /= counts[c];
                }
            }
        }

        free(counts);
    }

    free(assignments);
    return model;
}

Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X) {
    if (!model || !X || X->cols != model->centroids->cols) {
        return NULL;
    }

    Matrix *labels = matrix_alloc(X->rows, 1);
    if (!labels) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        labels->data[i] = find_nearest_centroid(&X->data[i * X->cols], model->centroids);
    }

    return labels;
}

void kmeans_free(KMeansModel *model) {
    if (model) {
        matrix_free(model->centroids);
        free(model);
    }
}
