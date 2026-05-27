/**
 * feature_selection_example.c - Feature Selection Demo
 *
 * Demonstrates:
 * - SelectKBest with different scoring functions
 * - VarianceThreshold for removing low-variance features
 * - Integration with pipeline and cross-validation
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "feature_selection.h"
#include "linear_regression.h"
#include "validation.h"
#include "utils.h"

// Generate data with some informative and some noisy features
static void generate_mixed_features(Matrix **X, Matrix **y, size_t n_samples) {
    size_t n_informative = 3;
    size_t n_noisy = 5;
    size_t n_features = n_informative + n_noisy;

    *X = matrix_alloc(n_samples, n_features);
    *y = matrix_alloc(n_samples, 1);

    rand_seed(42);

    for (size_t i = 0; i < n_samples; i++) {
        // Informative features (actually correlated with target)
        double x1 = rand_uniform_range(-5, 5);
        double x2 = rand_uniform_range(-3, 3);
        double x3 = rand_uniform_range(-2, 2);

        // Target is a function of informative features
        (*y)->data[i] = 2.0 * x1 + 1.5 * x2 - 0.5 * x3 + rand_normal_params(0, 0.5);

        // Store informative features
        (*X)->data[i * n_features + 0] = x1;
        (*X)->data[i * n_features + 1] = x2;
        (*X)->data[i * n_features + 2] = x3;

        // Noisy features (uncorrelated with target)
        for (size_t j = n_informative; j < n_features; j++) {
            (*X)->data[i * n_features + j] = rand_normal_params(0, 1);
        }
    }
}

// Generate classification data
static void generate_classification_mixed(Matrix **X, Matrix **y, size_t n_samples) {
    size_t n_features = 6;

    *X = matrix_alloc(n_samples, n_features);
    *y = matrix_alloc(n_samples, 1);

    rand_seed(123);

    for (size_t i = 0; i < n_samples; i++) {
        int label = (i < n_samples / 2) ? 0 : 1;
        (*y)->data[i] = label;

        // Features 0-1: discriminative
        (*X)->data[i * n_features + 0] = label * 2.0 + rand_normal_params(0, 0.5);
        (*X)->data[i * n_features + 1] = label * 1.5 + rand_normal_params(0, 0.5);

        // Features 2-3: noisy
        (*X)->data[i * n_features + 2] = rand_normal_params(0, 3);
        (*X)->data[i * n_features + 3] = rand_normal_params(0, 3);

        // Feature 4: constant (zero variance)
        (*X)->data[i * n_features + 4] = 5.0;

        // Feature 5: very low variance
        (*X)->data[i * n_features + 5] = 10.0 + rand_normal_params(0, 0.01);
    }
}

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - Feature Selection Demo\n");
    printf("==============================================\n\n");

    // Part 1: Regression with f_regression scoring
    printf("=== Part 1: Regression Feature Selection ===\n\n");

    printf("1. Generating regression data...\n");
    Matrix *X_reg, *y_reg;
    generate_mixed_features(&X_reg, &y_reg, 100);
    printf("   Generated %zu samples with %zu features\n", X_reg->rows, X_reg->cols);
    printf("   Features 0-2: informative\n");
    printf("   Features 3-7: noise\n\n");

    // F-regression scores
    printf("2. Computing F-regression scores...\n");
    double *f_scores = malloc(X_reg->cols * sizeof(double));
    f_regression(X_reg, y_reg, f_scores, NULL);

    printf("   Feature | F-score    | Type\n");
    printf("   --------|------------|-------------\n");
    for (size_t j = 0; j < X_reg->cols; j++) {
        const char *type = (j < 3) ? "informative" : "noise";
        printf("   %7zu | %10.4f | %s\n", j, f_scores[j], type);
    }
    printf("\n");
    free(f_scores);

    // SelectKBest for regression
    printf("3. SelectKBest (k=3) with f_regression...\n");
    SelectKBest *skb_reg = select_k_best_create(SCORE_F_REGRESSION, 3);
    skb_reg->base.fit((Estimator*)skb_reg, X_reg, y_reg);

    printf("   Selected features: ");
    const int *support = select_k_best_get_support(skb_reg);
    for (int j = 0; j < skb_reg->n_features_; j++) {
        if (support[j]) printf("%d ", j);
    }
    printf("\n\n");

    // Transform data
    Matrix *X_reg_reduced = skb_reg->base.transform((Estimator*)skb_reg, X_reg);
    printf("   Reduced from %zu to %zu features\n\n", X_reg->cols, X_reg_reduced->cols);

    // Compare model performance
    printf("4. Comparing model performance...\n");

    // Full features
    LinearRegression *lr_full = linear_regression_create(LINREG_SOLVER_CLOSED);
    CrossValResults *cv_full = cross_val_score((Estimator*)lr_full, X_reg, y_reg, 5, 1, 42);

    // Selected features
    LinearRegression *lr_selected = linear_regression_create(LINREG_SOLVER_CLOSED);
    CrossValResults *cv_selected = cross_val_score((Estimator*)lr_selected, X_reg_reduced, y_reg, 5, 1, 42);

    printf("   Full features (%zu):     R² = %.4f (+/- %.4f)\n",
           X_reg->cols, cv_full->mean_test_score, cv_full->std_test_score);
    printf("   Selected features (%d): R² = %.4f (+/- %.4f)\n\n",
           skb_reg->n_features_selected_, cv_selected->mean_test_score, cv_selected->std_test_score);

    lr_full->base.free((Estimator*)lr_full);
    lr_selected->base.free((Estimator*)lr_selected);
    cross_val_results_free(cv_full);
    cross_val_results_free(cv_selected);

    // Part 2: Classification with f_classif scoring
    printf("=== Part 2: Classification Feature Selection ===\n\n");

    printf("5. Generating classification data...\n");
    Matrix *X_clf, *y_clf;
    generate_classification_mixed(&X_clf, &y_clf, 100);
    printf("   Generated %zu samples with %zu features\n", X_clf->rows, X_clf->cols);
    printf("   Features 0-1: discriminative\n");
    printf("   Features 2-3: noise\n");
    printf("   Feature 4: constant\n");
    printf("   Feature 5: near-constant\n\n");

    // F-classif scores
    printf("6. Computing ANOVA F-values (f_classif)...\n");
    double *f_clf = malloc(X_clf->cols * sizeof(double));
    f_classif(X_clf, y_clf, f_clf, NULL);

    printf("   Feature | F-value\n");
    printf("   --------|----------\n");
    for (size_t j = 0; j < X_clf->cols; j++) {
        printf("   %7zu | %10.4f\n", j, f_clf[j]);
    }
    printf("\n");
    free(f_clf);

    // Mutual information
    printf("7. Computing Mutual Information...\n");
    double *mi_scores = malloc(X_clf->cols * sizeof(double));
    mutual_info_classif(X_clf, y_clf, mi_scores, 10);

    printf("   Feature | MI score\n");
    printf("   --------|----------\n");
    for (size_t j = 0; j < X_clf->cols; j++) {
        printf("   %7zu | %10.4f\n", j, mi_scores[j]);
    }
    printf("\n");
    free(mi_scores);

    // Part 3: VarianceThreshold
    printf("=== Part 3: Variance-based Selection ===\n\n");

    printf("8. Feature variances:\n");
    VarianceThreshold *vt = variance_threshold_create(0.1);
    vt->base.fit((Estimator*)vt, X_clf, NULL);

    const double *variances = variance_threshold_variances(vt);
    const int *vt_support = variance_threshold_get_support(vt);

    printf("   Feature | Variance   | Kept\n");
    printf("   --------|------------|------\n");
    for (int j = 0; j < vt->n_features_; j++) {
        printf("   %7d | %10.6f | %s\n", j, variances[j], vt_support[j] ? "Yes" : "No");
    }
    printf("\n");

    Matrix *X_var_selected = vt->base.transform((Estimator*)vt, X_clf);
    printf("   Kept %zu of %zu features (threshold=0.1)\n\n", X_var_selected->cols, X_clf->cols);

    // Print SelectKBest summary
    printf("9. SelectKBest Summary:\n");
    skb_reg->base.print_summary((Estimator*)skb_reg);

    // Cleanup
    printf("10. Cleaning up...\n");
    skb_reg->base.free((Estimator*)skb_reg);
    vt->base.free((Estimator*)vt);
    matrix_free(X_reg);
    matrix_free(y_reg);
    matrix_free(X_reg_reduced);
    matrix_free(X_clf);
    matrix_free(y_clf);
    matrix_free(X_var_selected);

    printf("    Done!\n\n");
    printf("==============================================\n");
    printf("  Feature selection demo completed!\n");
    printf("==============================================\n");

    return 0;
}
