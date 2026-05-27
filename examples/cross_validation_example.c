/**
 * cross_validation_example.c - Demonstrating cross-validation utilities
 *
 * Shows:
 * - K-Fold cross-validation
 * - cross_val_score() function
 * - Model comparison
 */

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "linear_regression.h"
#include "validation.h"
#include "utils.h"

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - Cross-Validation Demo\n");
    printf("==============================================\n\n");

    /* =========================================
     * Generate synthetic data
     * ========================================= */
    printf("1. Generating synthetic data...\n");

    size_t n_samples = 100;
    size_t n_features = 3;

    Matrix *X = matrix_alloc(n_samples, n_features);
    Matrix *y = matrix_alloc(n_samples, 1);

    rand_seed(42);
    for (size_t i = 0; i < n_samples; i++) {
        double x1 = rand_uniform_range(-5, 5);
        double x2 = rand_uniform_range(-5, 5);
        double x3 = rand_uniform_range(-5, 5);
        double noise = rand_normal() * 0.5;

        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);
        matrix_set(X, i, 2, x3);
        matrix_set(y, i, 0, 1.5 * x1 - 2.0 * x2 + 0.5 * x3 + 3.0 + noise);
    }
    printf("   Generated %zu samples with %zu features\n\n", n_samples, n_features);

    /* =========================================
     * 5-Fold Cross-Validation
     * ========================================= */
    printf("2. Running 5-Fold Cross-Validation...\n\n");

    // Create model
    LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

    // Run cross-validation
    CrossValResults *cv_results = cross_val_score(
        (Estimator*)model,
        X, y,
        5,      // 5 folds
        1,      // shuffle
        42      // seed
    );

    if (cv_results) {
        cross_val_results_print(cv_results);
        cross_val_results_free(cv_results);
    }

    /* =========================================
     * Compare different models
     * ========================================= */
    printf("3. Comparing Closed-form vs Gradient Descent...\n\n");

    // Closed-form
    LinearRegression *model_closed = linear_regression_create(LINREG_SOLVER_CLOSED);
    CrossValResults *cv_closed = cross_val_score(
        (Estimator*)model_closed, X, y, 5, 1, 42
    );

    // Gradient descent
    LinearRegression *model_gd = linear_regression_create_full(
        LINREG_SOLVER_GD, 0.01, 1000, 1e-6, 1, 0
    );
    CrossValResults *cv_gd = cross_val_score(
        (Estimator*)model_gd, X, y, 5, 1, 42
    );

    printf("   Model Comparison:\n");
    printf("   %-20s %-20s %-20s\n", "Model", "Mean Test Score", "Std");
    printf("   --------------------------------------------------------\n");

    if (cv_closed) {
        printf("   %-20s %-20.6f %-20.6f\n",
               "Closed-form", cv_closed->mean_test_score, cv_closed->std_test_score);
    }
    if (cv_gd) {
        printf("   %-20s %-20.6f %-20.6f\n",
               "Gradient Descent", cv_gd->mean_test_score, cv_gd->std_test_score);
    }
    printf("\n");

    /* =========================================
     * Manual K-Fold iteration
     * ========================================= */
    printf("4. Manual K-Fold iteration example...\n\n");

    KFold *kf = kfold_create(3, 1, 42);
    kfold_init(kf, X->rows);

    for (int fold = 0; fold < 3; fold++) {
        FoldIndices fi = kfold_get_fold(kf, fold);
        printf("   Fold %d: Train=%zu samples, Test=%zu samples\n",
               fold + 1, fi.n_train, fi.n_test);

        // Get actual data
        Matrix *X_train = matrix_get_rows(X, fi.train_indices, fi.n_train);
        Matrix *y_train = matrix_get_rows(y, fi.train_indices, fi.n_train);
        Matrix *X_test = matrix_get_rows(X, fi.test_indices, fi.n_test);
        Matrix *y_test = matrix_get_rows(y, fi.test_indices, fi.n_test);

        // Train a fresh model
        LinearRegression *fold_model = linear_regression_create(LINREG_SOLVER_CLOSED);
        fold_model->base.fit((Estimator*)fold_model, X_train, y_train);

        double train_score = fold_model->base.score((Estimator*)fold_model, X_train, y_train);
        double test_score = fold_model->base.score((Estimator*)fold_model, X_test, y_test);

        printf("           Train R²=%.4f, Test R²=%.4f\n", train_score, test_score);

        // Cleanup
        fold_model->base.free((Estimator*)fold_model);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        fold_indices_free(&fi);
    }

    kfold_free(kf);

    /* =========================================
     * Cleanup
     * ========================================= */
    printf("\n5. Cleaning up...\n");

    matrix_free(X);
    matrix_free(y);
    model->base.free((Estimator*)model);
    model_closed->base.free((Estimator*)model_closed);
    model_gd->base.free((Estimator*)model_gd);
    cross_val_results_free(cv_closed);
    cross_val_results_free(cv_gd);

    printf("   Done!\n");
    printf("\n==============================================\n");
    printf("  Cross-validation demo completed!\n");
    printf("==============================================\n");

    return 0;
}
