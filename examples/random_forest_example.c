/**
 * random_forest_example.c - Random Forest Demo
 *
 * Demonstrates:
 * - Random Forest Classifier with bootstrap sampling
 * - OOB (Out-of-Bag) score calculation
 * - Feature importance through ensemble averaging
 */

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "ensemble.h"
#include "decision_tree.h"
#include "validation.h"
#include "utils.h"

// Generate synthetic classification data
static void generate_classification_data(Matrix **X, Matrix **y, size_t n_samples) {
    *X = matrix_alloc(n_samples, 4);
    *y = matrix_alloc(n_samples, 1);

    rand_seed(42);

    for (size_t i = 0; i < n_samples; i++) {
        // Class 0: cluster around (0, 0)
        // Class 1: cluster around (3, 3)
        int label = (i < n_samples / 2) ? 0 : 1;

        double base_x = label * 3.0;
        double base_y = label * 3.0;

        (*X)->data[i * 4 + 0] = base_x + rand_normal();
        (*X)->data[i * 4 + 1] = base_y + rand_normal();
        // Noisy features
        (*X)->data[i * 4 + 2] = rand_normal_params(0, 5);
        (*X)->data[i * 4 + 3] = rand_normal_params(0, 5);

        (*y)->data[i] = label;
    }
}

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - Random Forest Demo\n");
    printf("  Ensemble learning with bootstrap aggregation\n");
    printf("==============================================\n\n");

    // 1. Generate data
    printf("1. Generating synthetic classification data...\n");
    Matrix *X, *y;
    generate_classification_data(&X, &y, 200);
    printf("   Generated %zu samples with %zu features\n", X->rows, X->cols);
    printf("   Features 0-1: informative, Features 2-3: noise\n\n");

    // 2. Split data
    printf("2. Splitting data (70%% train, 30%% test)...\n");
    size_t n_train = 140;
    size_t n_test = 60;

    size_t *train_idx = malloc(n_train * sizeof(size_t));
    size_t *test_idx = malloc(n_test * sizeof(size_t));

    rand_seed(42);
    // Simple shuffle for demo
    size_t *perm = malloc(X->rows * sizeof(size_t));
    for (size_t i = 0; i < X->rows; i++) perm[i] = i;
    for (size_t i = X->rows - 1; i > 0; i--) {
        size_t j = (size_t)(rand_uniform() * (i + 1));
        size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (size_t i = 0; i < n_train; i++) train_idx[i] = perm[i];
    for (size_t i = 0; i < n_test; i++) test_idx[i] = perm[n_train + i];
    free(perm);

    Matrix *X_train = matrix_get_rows(X, train_idx, n_train);
    Matrix *y_train = matrix_get_rows(y, train_idx, n_train);
    Matrix *X_test = matrix_get_rows(X, test_idx, n_test);
    Matrix *y_test = matrix_get_rows(y, test_idx, n_test);

    printf("   Train: %zu samples, Test: %zu samples\n\n", n_train, n_test);

    // 3. Single Decision Tree baseline
    printf("3. Training single Decision Tree (baseline)...\n");
    DecisionTreeClassifier *dt = decision_tree_classifier_create_full(
        CRITERION_GINI, 5, 2, 1, 0.0);

    dt->base.fit((Estimator*)dt, X_train, y_train);
    double dt_score = dt->base.score((Estimator*)dt, X_test, y_test);
    printf("   Decision Tree accuracy: %.4f\n\n", dt_score);

    // 4. Random Forest
    printf("4. Training Random Forest (50 trees)...\n");
    RandomForestClassifier *rf = random_forest_classifier_create_full(
        50,     // n_estimators
        5,      // max_depth
        2,      // min_samples_split
        1,      // min_samples_leaf
        0,      // max_features (0 = sqrt(n_features))
        1,      // bootstrap
        42      // seed
    );
    rf->base.verbose = VERBOSE_MINIMAL;

    rf->base.fit((Estimator*)rf, X_train, y_train);

    double rf_train_score = rf->base.score((Estimator*)rf, X_train, y_train);
    double rf_test_score = rf->base.score((Estimator*)rf, X_test, y_test);

    printf("   Train accuracy: %.4f\n", rf_train_score);
    printf("   Test accuracy:  %.4f\n", rf_test_score);
    printf("   OOB score:      %.4f\n\n", rf->oob_score_);

    // 5. Compare different forest sizes
    printf("5. Effect of number of trees...\n");
    printf("   n_trees | Train Acc | Test Acc  | OOB Score\n");
    printf("   --------|-----------|-----------|----------\n");

    int tree_counts[] = {5, 10, 25, 50, 100};
    for (int i = 0; i < 5; i++) {
        RandomForestClassifier *rf_test = random_forest_classifier_create_full(
            tree_counts[i], 5, 2, 1, 0, 1, 42);

        rf_test->base.fit((Estimator*)rf_test, X_train, y_train);

        double train_acc = rf_test->base.score((Estimator*)rf_test, X_train, y_train);
        double test_acc = rf_test->base.score((Estimator*)rf_test, X_test, y_test);

        printf("   %7d | %.4f    | %.4f    | %.4f\n",
               tree_counts[i], train_acc, test_acc, rf_test->oob_score_);

        rf_test->base.free((Estimator*)rf_test);
    }
    printf("\n");

    // 6. Probability predictions
    printf("6. Probability predictions (first 5 test samples)...\n");
    Matrix *proba = rf->base.predict_proba((Estimator*)rf, X_test);
    if (proba) {
        printf("   Sample | P(class=0) | P(class=1) | True | Pred\n");
        printf("   -------|------------|------------|------|-----\n");

        Matrix *preds = rf->base.predict((Estimator*)rf, X_test);
        for (size_t i = 0; i < 5 && i < X_test->rows; i++) {
            printf("   %6zu | %.4f     | %.4f     | %4d | %4d\n",
                   i,
                   proba->data[i * 2 + 0],
                   proba->data[i * 2 + 1],
                   (int)y_test->data[i],
                   (int)preds->data[i]);
        }
        matrix_free(preds);
        matrix_free(proba);
    }
    printf("\n");

    // 7. Cross-validation
    printf("7. 5-Fold Cross-validation...\n");
    Estimator *rf_cv = (Estimator*)random_forest_classifier_create_full(
        30, 5, 2, 1, 0, 1, 42);

    CrossValResults *cv = cross_val_score(rf_cv, X, y, 5, 1, 42);
    if (cv) {
        printf("   Fold scores: ");
        for (int i = 0; i < cv->n_splits; i++) {
            printf("%.4f ", cv->test_scores[i]);
        }
        printf("\n");
        printf("   Mean: %.4f (+/- %.4f)\n\n", cv->mean_test_score, cv->std_test_score);
        cross_val_results_free(cv);
    }
    rf_cv->free(rf_cv);

    // 8. Print summary
    printf("8. Model Summary:\n");
    rf->base.print_summary((Estimator*)rf);

    // Cleanup
    printf("9. Cleaning up...\n");
    dt->base.free((Estimator*)dt);
    rf->base.free((Estimator*)rf);
    matrix_free(X);
    matrix_free(y);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
    free(train_idx);
    free(test_idx);

    printf("   Done!\n\n");
    printf("==============================================\n");
    printf("  Random Forest demo completed!\n");
    printf("==============================================\n");

    return 0;
}
