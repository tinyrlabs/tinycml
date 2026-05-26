/**
 * @file logistic_regression_example.c
 * @brief Logistic regression demonstration for binary classification
 */

#include <stdio.h>
#include "matrix.h"
#include "csv.h"
#include "preprocessing.h"
#include "logistic_regression.h"
#include "metrics.h"

int main(void) {
    printf("=== Logistic Regression Example ===\n\n");

    /* Load data */
    Matrix *data = csv_load("data/binary_classification.csv", 1);
    if (!data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("Loaded %zu samples with %zu features\n", data->rows, data->cols - 1);

    /* Split into X and y */
    Matrix *X_raw = matrix_alloc(data->rows, 2);
    Matrix *y = matrix_alloc(data->rows, 1);

    for (size_t i = 0; i < data->rows; i++) {
        matrix_set(X_raw, i, 0, matrix_get(data, i, 0));
        matrix_set(X_raw, i, 1, matrix_get(data, i, 1));
        y->data[i] = matrix_get(data, i, 2);
    }

    /* Add bias column */
    Matrix *X = add_bias_column(X_raw);

    printf("Features shape: %zu x %zu (with bias)\n", X->rows, X->cols);
    printf("Target shape: %zu x %zu\n\n", y->rows, y->cols);

    /* Fit model */
    printf("Training logistic regression (lr=0.5, epochs=1000)...\n");
    Matrix *weights = logreg_fit(X, y, 0.5, 1000);

    if (!weights) {
        fprintf(stderr, "Training failed\n");
        matrix_free(data);
        matrix_free(X_raw);
        matrix_free(X);
        matrix_free(y);
        return 1;
    }

    printf("Weights:\n");
    printf("  Intercept: %.4f\n", matrix_get(weights, 0, 0));
    printf("  w1: %.4f\n", matrix_get(weights, 1, 0));
    printf("  w2: %.4f\n\n", matrix_get(weights, 2, 0));

    /* Make predictions */
    Matrix *proba = logreg_predict_proba(X, weights);
    Matrix *pred = logreg_predict(X, weights, 0.5);

    printf("Predictions:\n");
    printf("%-8s %-8s %-10s %-10s %-8s\n", "x1", "x2", "P(y=1)", "Predicted", "Actual");
    printf("----------------------------------------------\n");
    for (size_t i = 0; i < X->rows; i++) {
        printf("%-8.2f %-8.2f %-10.4f %-10.0f %-8.0f\n",
               matrix_get(X, i, 1),
               matrix_get(X, i, 2),
               proba->data[i],
               pred->data[i],
               y->data[i]);
    }

    /* Evaluate */
    printf("\nEvaluation Metrics:\n");
    printf("  Accuracy: %.2f%%\n", accuracy(y, pred) * 100);
    printf("  Precision: %.4f\n", precision(y, pred));
    printf("  Recall: %.4f\n", recall(y, pred));
    printf("  F1 Score: %.4f\n", f1_score(y, pred));

    ConfusionMatrix cm = confusion_matrix(y, pred);
    printf("\n");
    confusion_matrix_print(&cm);

    /* Cleanup */
    matrix_free(data);
    matrix_free(X_raw);
    matrix_free(X);
    matrix_free(y);
    matrix_free(weights);
    matrix_free(proba);
    matrix_free(pred);

    printf("\nDone!\n");
    return 0;
}
