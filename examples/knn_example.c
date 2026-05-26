/**
 * @file knn_example.c
 * @brief k-Nearest Neighbors demonstration
 */

#include <stdio.h>
#include "matrix.h"
#include "csv.h"
#include "knn.h"
#include "metrics.h"

int main(void) {
    printf("=== k-Nearest Neighbors Example ===\n\n");

    /* Load data */
    Matrix *data = csv_load("data/binary_classification.csv", 1);
    if (!data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("Loaded %zu samples\n", data->rows);

    /* Split into X and y */
    Matrix *X = matrix_alloc(data->rows, 2);
    Matrix *y = matrix_alloc(data->rows, 1);

    for (size_t i = 0; i < data->rows; i++) {
        matrix_set(X, i, 0, matrix_get(data, i, 0));
        matrix_set(X, i, 1, matrix_get(data, i, 1));
        y->data[i] = matrix_get(data, i, 2);
    }

    printf("Features shape: %zu x %zu\n", X->rows, X->cols);
    printf("Target shape: %zu x %zu\n\n", y->rows, y->cols);

    /* Fit k-NN model */
    int k = 3;
    printf("Fitting k-NN with k=%d...\n", k);
    KNNModel *model = knn_fit(X, y, k);

    if (!model) {
        fprintf(stderr, "Training failed\n");
        matrix_free(data);
        matrix_free(X);
        matrix_free(y);
        return 1;
    }

    /* Predict on training data */
    Matrix *pred = knn_predict(model, X);

    printf("\nPredictions on training data:\n");
    printf("%-8s %-8s %-10s %-8s\n", "x1", "x2", "Predicted", "Actual");
    printf("----------------------------------\n");
    for (size_t i = 0; i < X->rows; i++) {
        printf("%-8.2f %-8.2f %-10.0f %-8.0f\n",
               matrix_get(X, i, 0),
               matrix_get(X, i, 1),
               pred->data[i],
               y->data[i]);
    }

    /* Evaluate */
    printf("\nTraining Accuracy: %.2f%%\n", accuracy(y, pred) * 100);

    /* Predict on new points */
    printf("\nPredicting new points:\n");
    Matrix *X_new = matrix_alloc(3, 2);
    matrix_set(X_new, 0, 0, 2.0); matrix_set(X_new, 0, 1, 2.0);  /* Should be class 0 */
    matrix_set(X_new, 1, 0, 3.5); matrix_set(X_new, 1, 1, 3.5);  /* Should be class 1 */
    matrix_set(X_new, 2, 0, 2.5); matrix_set(X_new, 2, 1, 3.0);  /* Boundary case */

    Matrix *pred_new = knn_predict(model, X_new);

    printf("%-8s %-8s %-10s\n", "x1", "x2", "Predicted");
    printf("--------------------------\n");
    for (size_t i = 0; i < X_new->rows; i++) {
        printf("%-8.2f %-8.2f %-10.0f\n",
               matrix_get(X_new, i, 0),
               matrix_get(X_new, i, 1),
               pred_new->data[i]);
    }

    /* Cleanup */
    matrix_free(data);
    matrix_free(X);
    matrix_free(y);
    matrix_free(pred);
    matrix_free(X_new);
    matrix_free(pred_new);
    knn_free(model);

    printf("\nDone!\n");
    return 0;
}
