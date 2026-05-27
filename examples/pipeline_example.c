/**
 * pipeline_example.c - Demonstrating Pipeline functionality
 *
 * Shows scikit-learn style pipelines:
 * - Chaining transformers and estimators
 * - StandardScaler + LinearRegression
 * - PolynomialFeatures for non-linear regression
 */

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "pipeline.h"
#include "linear_regression.h"
#include "validation.h"
#include "utils.h"

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - Pipeline Demo\n");
    printf("  scikit-learn style pipelines in pure C\n");
    printf("==============================================\n\n");

    /* =========================================
     * Generate synthetic data with different scales
     * ========================================= */
    printf("1. Generating data with different feature scales...\n");

    size_t n_samples = 100;
    Matrix *X = matrix_alloc(n_samples, 2);
    Matrix *y = matrix_alloc(n_samples, 1);

    rand_seed(42);
    for (size_t i = 0; i < n_samples; i++) {
        // Feature 1: small scale (-1 to 1)
        double x1 = rand_uniform_range(-1, 1);
        // Feature 2: large scale (-1000 to 1000)
        double x2 = rand_uniform_range(-1000, 1000);
        double noise = rand_normal() * 5;

        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);
        // y depends equally on both features
        matrix_set(y, i, 0, 100 * x1 + 0.1 * x2 + noise);
    }

    printf("   Feature 1 range: [-1, 1]\n");
    printf("   Feature 2 range: [-1000, 1000]\n\n");

    /* =========================================
     * Without scaling - gradient descent struggles
     * ========================================= */
    printf("2. Without scaling (gradient descent)...\n");

    LinearRegression *model_no_scale = linear_regression_create_full(
        LINREG_SOLVER_GD, 0.0001, 1000, 1e-6, 1, 0
    );

    CrossValResults *cv_no_scale = cross_val_score(
        (Estimator*)model_no_scale, X, y, 5, 1, 42
    );

    if (cv_no_scale) {
        printf("   Mean R²: %.6f (+/- %.6f)\n",
               cv_no_scale->mean_test_score, cv_no_scale->std_test_score * 2);
    }

    /* =========================================
     * With Pipeline: StandardScaler + LinearRegression
     * ========================================= */
    printf("\n3. With Pipeline (StandardScaler + LinearRegression)...\n");

    // Create pipeline
    Pipeline *pipe = pipeline_create();

    // Add scaler
    pipeline_add_transformer(pipe, "scaler",
        (Transformer*)standard_scaler_transformer_create());

    // Add model
    pipeline_add_estimator(pipe, "model",
        (Estimator*)linear_regression_create_full(
            LINREG_SOLVER_GD, 0.01, 1000, 1e-6, 1, 0
        ));

    // Print pipeline structure
    pipeline_print_summary((Estimator*)pipe);

    // Cross-validate the pipeline
    CrossValResults *cv_pipe = cross_val_score(
        (Estimator*)pipe, X, y, 5, 1, 42
    );

    if (cv_pipe) {
        printf("   Mean R²: %.6f (+/- %.6f)\n",
               cv_pipe->mean_test_score, cv_pipe->std_test_score * 2);
    }

    /* =========================================
     * Polynomial Features for non-linear data
     * ========================================= */
    printf("\n4. Polynomial regression with Pipeline...\n");

    // Generate non-linear data: y = x^2 + noise
    Matrix *X_poly = matrix_alloc(n_samples, 1);
    Matrix *y_poly = matrix_alloc(n_samples, 1);

    for (size_t i = 0; i < n_samples; i++) {
        double x = rand_uniform_range(-3, 3);
        double noise = rand_normal() * 0.5;
        matrix_set(X_poly, i, 0, x);
        matrix_set(y_poly, i, 0, x * x + noise);
    }

    // Linear regression on non-linear data
    LinearRegression *linear_model = linear_regression_create(LINREG_SOLVER_CLOSED);
    CrossValResults *cv_linear = cross_val_score(
        (Estimator*)linear_model, X_poly, y_poly, 5, 1, 42
    );
    printf("   Linear model on x² data:\n");
    printf("   Mean R²: %.6f\n", cv_linear ? cv_linear->mean_test_score : 0);

    // Polynomial pipeline
    Pipeline *poly_pipe = pipeline_create();
    pipeline_add_transformer(poly_pipe, "poly",
        (Transformer*)polynomial_features_create(2, 0, 0));  // degree=2, no bias, not interaction only
    pipeline_add_estimator(poly_pipe, "model",
        (Estimator*)linear_regression_create(LINREG_SOLVER_CLOSED));

    CrossValResults *cv_poly = cross_val_score(
        (Estimator*)poly_pipe, X_poly, y_poly, 5, 1, 42
    );
    printf("   Polynomial (degree=2) model on x² data:\n");
    printf("   Mean R²: %.6f\n", cv_poly ? cv_poly->mean_test_score : 0);

    /* =========================================
     * Full pipeline: Scale + Polynomial + Model
     * ========================================= */
    printf("\n5. Full pipeline: Scaler + Polynomial + Model...\n");

    Pipeline *full_pipe = pipeline_create();
    pipeline_add_transformer(full_pipe, "scaler",
        (Transformer*)standard_scaler_transformer_create());
    pipeline_add_transformer(full_pipe, "poly",
        (Transformer*)polynomial_features_create(2, 0, 0));
    pipeline_add_estimator(full_pipe, "model",
        (Estimator*)linear_regression_create(LINREG_SOLVER_CLOSED));

    pipeline_print_summary((Estimator*)full_pipe);

    CrossValResults *cv_full = cross_val_score(
        (Estimator*)full_pipe, X_poly, y_poly, 5, 1, 42
    );
    printf("   Mean R²: %.6f\n", cv_full ? cv_full->mean_test_score : 0);

    /* =========================================
     * Cleanup
     * ========================================= */
    printf("\n6. Cleaning up...\n");

    matrix_free(X);
    matrix_free(y);
    matrix_free(X_poly);
    matrix_free(y_poly);
    model_no_scale->base.free((Estimator*)model_no_scale);
    linear_model->base.free((Estimator*)linear_model);
    pipeline_free((Estimator*)pipe);
    pipeline_free((Estimator*)poly_pipe);
    pipeline_free((Estimator*)full_pipe);
    cross_val_results_free(cv_no_scale);
    cross_val_results_free(cv_pipe);
    cross_val_results_free(cv_linear);
    cross_val_results_free(cv_poly);
    cross_val_results_free(cv_full);

    printf("   Done!\n");
    printf("\n==============================================\n");
    printf("  Pipeline demo completed!\n");
    printf("==============================================\n");

    return 0;
}
