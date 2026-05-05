/**
 * @file test_model_selection.c
 * @brief Unit tests for model selection (GridSearchCV, ParameterGrid)
 */

#include "test_harness.h"
#include "matrix.h"
#include "model_selection.h"
#include "decision_tree.h"
#include <math.h>

TEST(test_param_grid_init) {
    /* Create grid with 2 params: one int, one double */
    ParameterGrid *grid = param_grid_create(2);
    ASSERT_NOT_NULL(grid);
    ASSERT_EQ(grid->n_params, 2);

    /* Add integer param: max_depth = [3, 5, 7] */
    int depth_vals[] = {3, 5, 7};
    int rc = param_grid_add_int(grid, 0, "max_depth", depth_vals, 3);
    ASSERT_EQ(rc, 0);

    /* Add double param: min_impurity_decrease = [0.0, 0.01] */
    double imp_vals[] = {0.0, 0.01};
    rc = param_grid_add_double(grid, 1, "min_impurity_decrease", imp_vals, 2);
    ASSERT_EQ(rc, 0);

    /* Verify total combinations: 3 * 2 = 6 */
    int n_combos = param_grid_get_n_combinations(grid);
    ASSERT_EQ(n_combos, 6);

    /* Verify first combination */
    double *combo = param_grid_get_combination(grid, 0);
    ASSERT_NOT_NULL(combo);
    ASSERT_EQ((int)combo[0], 3);
    ASSERT_NEAR(combo[1], 0.0, 1e-9);
    free(combo);

    /* Verify last combination */
    combo = param_grid_get_combination(grid, 5);
    ASSERT_NOT_NULL(combo);
    ASSERT_EQ((int)combo[0], 7);
    ASSERT_NEAR(combo[1], 0.01, 1e-9);
    free(combo);

    param_grid_free(grid);
}

TEST(test_grid_search_create_free) {
    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);

    ParameterGrid *grid = param_grid_create(1);
    ASSERT_NOT_NULL(grid);

    int depth_vals[] = {3, 5};
    param_grid_add_int(grid, 0, "max_depth", depth_vals, 2);

    GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, grid, 3);
    ASSERT_NOT_NULL(gs);
    ASSERT_EQ(gs->cv, 3);
    ASSERT_EQ(gs->n_results, 2);

    grid_search_cv_free(gs);
    /* Note: grid_search_cv_free does NOT free the base_estimator or param_grid.
       They need to be freed separately. */
    param_grid_free(grid);
    decision_tree_classifier_free((Estimator*)dt);
}

TEST(test_grid_search_fit) {
    /* Create a simple classification dataset */
    int n = 30;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    /* Two separable clusters */
    for (int i = 0; i < n / 2; i++) {
        matrix_set(X, i, 0, 1.0 + (i % 5) * 0.1);
        matrix_set(X, i, 1, 1.0 + (i % 3) * 0.2);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = n / 2; i < n; i++) {
        matrix_set(X, i, 0, 5.0 + (i % 5) * 0.1);
        matrix_set(X, i, 1, 5.0 + (i % 3) * 0.2);
        matrix_set(y, i, 0, 1.0);
    }

    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);

    ParameterGrid *grid = param_grid_create(1);
    int depth_vals[] = {3, 5, 7};
    param_grid_add_int(grid, 0, "max_depth", depth_vals, 3);

    GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, grid, 3);
    ASSERT_NOT_NULL(gs);

    /* Suppress verbose output during tests */
    gs->verbose = 0;

    int rc = grid_search_cv_fit(gs, X, y);
    ASSERT_EQ(rc, 0);

    /* Best score should be > 0.5 for separable data */
    double best_score = grid_search_cv_best_score(gs);
    ASSERT(best_score > 0.5);

    grid_search_cv_free(gs);
    param_grid_free(grid);
    decision_tree_classifier_free((Estimator*)dt);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_grid_search_best_params) {
    /* Verify best params are set after fit */
    int n = 24;
    Matrix *X = matrix_alloc(n, 2);
    Matrix *y = matrix_alloc(n, 1);

    for (int i = 0; i < n / 2; i++) {
        matrix_set(X, i, 0, 0.0 + i * 0.5);
        matrix_set(X, i, 1, 0.0 + i * 0.5);
        matrix_set(y, i, 0, 0.0);
    }
    for (int i = n / 2; i < n; i++) {
        matrix_set(X, i, 0, 10.0 + (i - n / 2) * 0.5);
        matrix_set(X, i, 1, 10.0 + (i - n / 2) * 0.5);
        matrix_set(y, i, 0, 1.0);
    }

    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);

    ParameterGrid *grid = param_grid_create(1);
    int depth_vals[] = {3, 5, 7};
    param_grid_add_int(grid, 0, "max_depth", depth_vals, 3);

    GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, grid, 3);
    ASSERT_NOT_NULL(gs);
    gs->verbose = 0;

    int rc = grid_search_cv_fit(gs, X, y);
    ASSERT_EQ(rc, 0);

    /* best_params should not be NULL */
    const double *best_params = grid_search_cv_best_params(gs);
    ASSERT_NOT_NULL(best_params);

    /* The param value should be one of [3, 5, 7] */
    int best_depth = (int)best_params[0];
    ASSERT(best_depth == 3 || best_depth == 5 || best_depth == 7);

    /* best_estimator should be set (refit on all data) */
    Estimator *best_est = grid_search_cv_best_estimator(gs);
    ASSERT_NOT_NULL(best_est);
    ASSERT_EQ(best_est->is_fitted, 1);

    grid_search_cv_free(gs);
    param_grid_free(grid);
    decision_tree_classifier_free((Estimator*)dt);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Model Selection Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_param_grid_init);
    RUN_TEST(test_grid_search_create_free);
    RUN_TEST(test_grid_search_fit);
    RUN_TEST(test_grid_search_best_params);

    TEST_SUMMARY();
}
