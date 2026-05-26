/**
 * @file linear_regression_example.c
 * @brief Linear regression demonstration
 */

#include <stdio.h>
#include "matrix.h"
#include "csv.h"
#include "preprocessing.h"
#include "linear_regression.h"
#include "metrics.h"

int main(void) {
    printf("=== Linear Regression Example ===\n\n");

    /* Load data */
    Matrix *data = csv_load("data/simple_linear.csv", 1);
    if (!data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("Loaded %zu samples\n", data->rows);

    /* Split into X and y */
    Matrix *X_raw = matrix_alloc(data->rows, 1);
    Matrix *y = matrix_alloc(data->rows, 1);

    for (size_t i = 0; i < data->rows; i++) {
        X_raw->data[i] = matrix_get(data, i, 0);
        y->data[i] = matrix_get(data, i, 1);
    }

    /* Add bias column */
    Matrix *X = add_bias_column(X_raw);

    printf("Features shape: %zu x %zu\n", X->rows, X->cols);
    printf("Target shape: %zu x %zu\n\n", y->rows, y->cols);

    /* Fit model using closed-form solution */
    printf("Fitting with closed-form solution (normal equation)...\n");
    Matrix *weights_closed = linreg_fit_closed(X, y);

    if (weights_closed) {
        printf("Weights (closed-form):\n");
        printf("  Intercept: %.4f\n", matrix_get(weights_closed, 0, 0));
        printf("  Slope: %.4f\n", matrix_get(weights_closed, 1, 0));

        Matrix *pred_closed = linreg_predict(X, weights_closed);
        printf("  MSE: %.6f\n", mse(y, pred_closed));
        printf("  RMSE: %.6f\n\n", rmse(y, pred_closed));
        matrix_free(pred_closed);
    }

    /* Fit model using gradient descent */
    printf("Fitting with gradient descent (lr=0.01, epochs=1000)...\n");
    Matrix *weights_gd = linreg_fit_gd(X, y, 0.01, 1000);

    if (weights_gd) {
        printf("Weights (gradient descent):\n");
        printf("  Intercept: %.4f\n", matrix_get(weights_gd, 0, 0));
        printf("  Slope: %.4f\n", matrix_get(weights_gd, 1, 0));

        Matrix *pred_gd = linreg_predict(X, weights_gd);
        printf("  MSE: %.6f\n", mse(y, pred_gd));
        printf("  RMSE: %.6f\n\n", rmse(y, pred_gd));
        matrix_free(pred_gd);
    }

    /* Make predictions on new data */
    printf("Predictions for new values:\n");
    double new_x[] = {11.0, 12.0, 15.0};
    for (int i = 0; i < 3; i++) {
        double pred_y = matrix_get(weights_closed, 0, 0) +
                        matrix_get(weights_closed, 1, 0) * new_x[i];
        printf("  x=%.1f -> y=%.2f\n", new_x[i], pred_y);
    }

    /* Cleanup */
    matrix_free(data);
    matrix_free(X_raw);
    matrix_free(X);
    matrix_free(y);
    matrix_free(weights_closed);
    matrix_free(weights_gd);

    printf("\nDone!\n");
    return 0;
}
