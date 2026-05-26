/**
 * estimator_api_example.c - Demonstrating scikit-learn style API
 *
 * Shows how to use the unified Estimator interface:
 * - Create model with linear_regression_create()
 * - Train with model->base.fit()
 * - Predict with model->base.predict()
 * - Score with model->base.score()
 * - Save/Load models
 */

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "linear_regression.h"
#include "preprocessing.h"
#include "csv.h"
#include "utils.h"

/* Custom callback to track training progress */
void my_callback(int epoch, double loss, double metric, void *user_data) {
    int *best_epoch = (int*)user_data;
    if (epoch == 1 || epoch % 100 == 0) {
        printf("  [Callback] Epoch %d: loss=%.6f, R²=%.6f\n", epoch, loss, metric);
    }
    *best_epoch = epoch;
}

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - Estimator API Demo\n");
    printf("  scikit-learn style interface in pure C\n");
    printf("==============================================\n\n");

    /* =========================================
     * Generate synthetic data: y = 2*x1 + 3*x2 + 5 + noise
     * ========================================= */
    printf("1. Generating synthetic data...\n");

    size_t n_samples = 100;
    size_t n_features = 2;

    Matrix *X = matrix_alloc(n_samples, n_features);
    Matrix *y = matrix_alloc(n_samples, 1);

    rand_seed(42);
    for (size_t i = 0; i < n_samples; i++) {
        double x1 = rand_uniform_range(-10, 10);
        double x2 = rand_uniform_range(-10, 10);
        double noise = rand_normal() * 0.5;

        matrix_set(X, i, 0, x1);
        matrix_set(X, i, 1, x2);
        matrix_set(y, i, 0, 2.0 * x1 + 3.0 * x2 + 5.0 + noise);
    }
    printf("   Generated %zu samples with %zu features\n\n", n_samples, n_features);

    /* =========================================
     * Train/Test Split
     * ========================================= */
    printf("2. Splitting data (80%% train, 20%% test)...\n");
    TrainTestSplit split = train_test_split(X, y, 0.2, 42);
    printf("   Train: %zu samples, Test: %zu samples\n\n",
           split.X_train->rows, split.X_test->rows);

    /* =========================================
     * Method 1: Closed-form solution (like sklearn default)
     * ========================================= */
    printf("3. Training with closed-form solution...\n");

    // Create model - like sklearn: LinearRegression()
    LinearRegression *model1 = linear_regression_create(LINREG_SOLVER_CLOSED);
    estimator_set_verbose((Estimator*)model1, VERBOSE_MINIMAL);

    // Fit - like sklearn: model.fit(X, y)
    model1->base.fit((Estimator*)model1, split.X_train, split.y_train);

    // Score - like sklearn: model.score(X, y)
    double train_score = model1->base.score((Estimator*)model1, split.X_train, split.y_train);
    double test_score = model1->base.score((Estimator*)model1, split.X_test, split.y_test);

    printf("   Train R²: %.6f\n", train_score);
    printf("   Test R²:  %.6f\n", test_score);

    // Print model summary
    model1->base.print_summary((Estimator*)model1);

    /* =========================================
     * Method 2: Gradient descent with callback
     * ========================================= */
    printf("4. Training with gradient descent...\n");

    // Create with custom hyperparameters
    LinearRegression *model2 = linear_regression_create_full(
        LINREG_SOLVER_GD,   // solver
        0.01,               // learning_rate
        500,                // max_iter
        1e-6,               // tolerance
        1,                  // fit_intercept
        0                   // normalize
    );

    // Set verbose and callback
    estimator_set_verbose((Estimator*)model2, VERBOSE_PROGRESS);
    int best_epoch = 0;
    estimator_set_callback((Estimator*)model2, my_callback, &best_epoch);

    // Fit
    model2->base.fit((Estimator*)model2, split.X_train, split.y_train);

    // Score
    double gd_test_score = model2->base.score((Estimator*)model2, split.X_test, split.y_test);
    printf("   Final test R²: %.6f (converged at epoch %d)\n", gd_test_score, best_epoch);

    // Print summary
    model2->base.print_summary((Estimator*)model2);

    /* =========================================
     * Model Serialization (save/load)
     * ========================================= */
    printf("5. Testing model serialization...\n");

    // Save model
    const char *model_file = "model.tcml";
    int save_result = model1->base.save((Estimator*)model1, model_file);
    printf("   Save result: %s\n", save_result == 0 ? "Success" : "Failed");

    // Load model
    Estimator *loaded = linear_regression_load(model_file);
    if (loaded) {
        double loaded_score = loaded->score(loaded, split.X_test, split.y_test);
        printf("   Loaded model test R²: %.6f\n", loaded_score);
        loaded->free(loaded);
    }

    // Clean up model file
    remove(model_file);

    /* =========================================
     * Predictions
     * ========================================= */
    printf("\n6. Making predictions...\n");

    // Create a new sample
    Matrix *X_new = matrix_alloc(1, 2);
    matrix_set(X_new, 0, 0, 5.0);   // x1 = 5
    matrix_set(X_new, 0, 1, 3.0);   // x2 = 3
    // Expected: y = 2*5 + 3*3 + 5 = 24

    Matrix *y_pred = model1->base.predict((Estimator*)model1, X_new);
    printf("   Input: x1=5.0, x2=3.0\n");
    printf("   Expected: 2*5 + 3*3 + 5 = 24.0\n");
    printf("   Predicted: %.4f\n", matrix_get(y_pred, 0, 0));

    /* =========================================
     * Clone model (for cross-validation later)
     * ========================================= */
    printf("\n7. Cloning model...\n");
    Estimator *cloned = model1->base.clone((Estimator*)model1);
    if (cloned) {
        printf("   Clone created (unfitted)\n");
        printf("   Cloned model fitted: %s\n", cloned->is_fitted ? "Yes" : "No");

        // Fit the clone
        cloned->fit(cloned, split.X_train, split.y_train);
        double clone_score = cloned->score(cloned, split.X_test, split.y_test);
        printf("   Clone trained, test R²: %.6f\n", clone_score);
        cloned->free(cloned);
    }

    /* =========================================
     * Cleanup
     * ========================================= */
    printf("\n8. Cleaning up...\n");

    matrix_free(X);
    matrix_free(y);
    matrix_free(X_new);
    matrix_free(y_pred);
    train_test_split_free(&split);
    model1->base.free((Estimator*)model1);
    model2->base.free((Estimator*)model2);

    printf("   Done!\n");
    printf("\n==============================================\n");
    printf("  Demo completed successfully!\n");
    printf("==============================================\n");

    return 0;
}
