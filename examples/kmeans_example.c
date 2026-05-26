/**
 * @file kmeans_example.c
 * @brief k-Means clustering demonstration
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "csv.h"
#include "kmeans.h"

int main(void) {
    printf("=== k-Means Clustering Example ===\n\n");

    /* Load data */
    Matrix *data = csv_load("data/clusters.csv", 1);
    if (!data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("Loaded %zu samples\n", data->rows);
    printf("Data shape: %zu x %zu\n\n", data->rows, data->cols);

    /* Fit k-Means */
    int k = 3;
    int max_iter = 100;
    unsigned int seed = 42;

    printf("Fitting k-Means with k=%d, max_iter=%d...\n", k, max_iter);
    KMeansModel *model = kmeans_fit(data, k, max_iter, seed);

    if (!model) {
        fprintf(stderr, "Clustering failed\n");
        matrix_free(data);
        return 1;
    }

    /* Print centroids */
    printf("\nCluster Centroids:\n");
    for (int c = 0; c < k; c++) {
        printf("  Cluster %d: (%.2f, %.2f)\n", c,
               matrix_get(model->centroids, c, 0),
               matrix_get(model->centroids, c, 1));
    }

    /* Predict cluster assignments */
    Matrix *labels = kmeans_predict(model, data);

    printf("\nCluster Assignments:\n");
    printf("%-8s %-8s %-8s\n", "x1", "x2", "Cluster");
    printf("------------------------\n");
    for (size_t i = 0; i < data->rows; i++) {
        printf("%-8.2f %-8.2f %-8.0f\n",
               matrix_get(data, i, 0),
               matrix_get(data, i, 1),
               labels->data[i]);
    }

    /* Count samples per cluster */
    int *counts = calloc(k, sizeof(int));
    for (size_t i = 0; i < data->rows; i++) {
        counts[(int)labels->data[i]]++;
    }

    printf("\nSamples per cluster:\n");
    for (int c = 0; c < k; c++) {
        printf("  Cluster %d: %d samples\n", c, counts[c]);
    }
    free(counts);

    /* Predict on new points */
    printf("\nPredicting new points:\n");
    Matrix *X_new = matrix_alloc(3, 2);
    matrix_set(X_new, 0, 0, 1.0); matrix_set(X_new, 0, 1, 1.0);
    matrix_set(X_new, 1, 0, 5.0); matrix_set(X_new, 1, 1, 5.0);
    matrix_set(X_new, 2, 0, 9.0); matrix_set(X_new, 2, 1, 1.0);

    Matrix *labels_new = kmeans_predict(model, X_new);

    printf("%-8s %-8s %-8s\n", "x1", "x2", "Cluster");
    printf("------------------------\n");
    for (size_t i = 0; i < X_new->rows; i++) {
        printf("%-8.2f %-8.2f %-8.0f\n",
               matrix_get(X_new, i, 0),
               matrix_get(X_new, i, 1),
               labels_new->data[i]);
    }

    /* Cleanup */
    matrix_free(data);
    matrix_free(labels);
    matrix_free(X_new);
    matrix_free(labels_new);
    kmeans_free(model);

    printf("\nDone!\n");
    return 0;
}
