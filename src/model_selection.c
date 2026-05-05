/**
 * model_selection.c - GridSearchCV and learning curves implementation
 */

#include "model_selection.h"
#include "linear_regression.h"
#include "decision_tree.h"
#include "neural_network.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Parameter Grid Implementation
 * ============================================ */

ParameterGrid* param_grid_create(int n_params) {
    ParameterGrid *grid = calloc(1, sizeof(ParameterGrid));
    if (!grid) return NULL;

    grid->params = calloc(n_params, sizeof(ParamSpec));
    if (!grid->params) {
        free(grid);
        return NULL;
    }
    grid->n_params = n_params;

    return grid;
}

int param_grid_add_int(ParameterGrid *grid, int index, const char *name,
                       const int *values, int n_values) {
    if (!grid || index < 0 || index >= grid->n_params) return -1;

    ParamSpec *spec = &grid->params[index];
    spec->name = name;
    spec->type = PARAM_INT;
    spec->data.int_vals.values = malloc(n_values * sizeof(int));
    if (!spec->data.int_vals.values) return -1;

    memcpy(spec->data.int_vals.values, values, n_values * sizeof(int));
    spec->data.int_vals.n_values = n_values;

    return 0;
}

int param_grid_add_double(ParameterGrid *grid, int index, const char *name,
                          const double *values, int n_values) {
    if (!grid || index < 0 || index >= grid->n_params) return -1;

    ParamSpec *spec = &grid->params[index];
    spec->name = name;
    spec->type = PARAM_DOUBLE;
    spec->data.double_vals.values = malloc(n_values * sizeof(double));
    if (!spec->data.double_vals.values) return -1;

    memcpy(spec->data.double_vals.values, values, n_values * sizeof(double));
    spec->data.double_vals.n_values = n_values;

    return 0;
}

int param_grid_get_n_combinations(const ParameterGrid *grid) {
    if (!grid || grid->n_params == 0) return 0;

    int n_combos = 1;
    for (int i = 0; i < grid->n_params; i++) {
        ParamSpec *spec = &grid->params[i];
        switch (spec->type) {
            case PARAM_INT:
                n_combos *= spec->data.int_vals.n_values;
                break;
            case PARAM_DOUBLE:
                n_combos *= spec->data.double_vals.n_values;
                break;
            case PARAM_ENUM:
                n_combos *= spec->data.enum_vals.n_values;
                break;
        }
    }
    return n_combos;
}

double* param_grid_get_combination(const ParameterGrid *grid, int combo_index) {
    if (!grid) return NULL;

    double *params = malloc(grid->n_params * sizeof(double));
    if (!params) return NULL;

    int divisor = 1;
    for (int i = grid->n_params - 1; i >= 0; i--) {
        ParamSpec *spec = &grid->params[i];
        int n_values = 0;

        switch (spec->type) {
            case PARAM_INT:
                n_values = spec->data.int_vals.n_values;
                break;
            case PARAM_DOUBLE:
                n_values = spec->data.double_vals.n_values;
                break;
            case PARAM_ENUM:
                n_values = spec->data.enum_vals.n_values;
                break;
        }

        int idx = (combo_index / divisor) % n_values;
        divisor *= n_values;

        switch (spec->type) {
            case PARAM_INT:
                params[i] = (double)spec->data.int_vals.values[idx];
                break;
            case PARAM_DOUBLE:
                params[i] = spec->data.double_vals.values[idx];
                break;
            case PARAM_ENUM:
                params[i] = (double)spec->data.enum_vals.values[idx];
                break;
        }
    }

    return params;
}

void param_grid_free(ParameterGrid *grid) {
    if (!grid) return;

    for (int i = 0; i < grid->n_params; i++) {
        ParamSpec *spec = &grid->params[i];
        switch (spec->type) {
            case PARAM_INT:
                free(spec->data.int_vals.values);
                break;
            case PARAM_DOUBLE:
                free(spec->data.double_vals.values);
                break;
            case PARAM_ENUM:
                free(spec->data.enum_vals.values);
                break;
        }
    }
    free(grid->params);
    free(grid);
}

/* ============================================
 * GridSearchCV Implementation
 * ============================================ */

GridSearchCV* grid_search_cv_create(
    Estimator *estimator,
    ParameterGrid *param_grid,
    int cv
) {
    GridSearchCV *gs = calloc(1, sizeof(GridSearchCV));
    if (!gs) return NULL;

    gs->base_estimator = estimator;
    gs->param_grid = param_grid;
    gs->cv = cv;
    gs->shuffle = 1;
    gs->seed = 42;
    gs->verbose = 1;
    gs->refit = 1;

    gs->is_classification = (estimator->task == TASK_CLASSIFICATION);

    int n_combos = param_grid_get_n_combinations(param_grid);
    gs->results = calloc(n_combos, sizeof(GridSearchResult));
    gs->n_results = n_combos;
    gs->best_index = -1;
    gs->best_score = -INFINITY;
    gs->best_estimator = NULL;
    gs->best_params = NULL;

    return gs;
}

// Helper: Apply parameter to LinearRegression model
static void apply_linreg_params(LinearRegression *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "learning_rate") == 0) {
            model->learning_rate = value;
        } else if (strcmp(name, "max_iter") == 0) {
            model->max_iter = (int)value;
        } else if (strcmp(name, "tol") == 0) {
            model->tol = value;
        } else if (strcmp(name, "solver") == 0) {
            model->solver = (LinRegSolver)(int)value;
        }
    }
}

// Helper: Apply parameter to DecisionTreeClassifier model
static void apply_dt_params(DecisionTreeClassifier *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "max_depth") == 0) {
            model->max_depth = (int)value;
        } else if (strcmp(name, "min_samples_split") == 0) {
            model->min_samples_split = (int)value;
        } else if (strcmp(name, "min_samples_leaf") == 0) {
            model->min_samples_leaf = (int)value;
        } else if (strcmp(name, "min_impurity_decrease") == 0) {
            model->min_impurity_decrease = value;
        } else if (strcmp(name, "max_features") == 0) {
            model->max_features = (int)value;
        }
    }
}

// Helper: Apply parameter to NeuralNetwork model
static void apply_nn_params(NeuralNetwork *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "learning_rate") == 0) {
            model->learning_rate = value;
        } else if (strcmp(name, "max_epochs") == 0) {
            model->max_epochs = (int)value;
        } else if (strcmp(name, "batch_size") == 0) {
            model->batch_size = (int)value;
        } else if (strcmp(name, "momentum") == 0) {
            model->momentum = value;
        } else if (strcmp(name, "l2_reg") == 0) {
            model->l2_reg = value;
        } else if (strcmp(name, "dropout_rate") == 0) {
            model->dropout_rate = value;
        }
    }
}

// Helper: Apply parameters to KNN model (k is set at create time, not tunable via GridSearch)
// KNN uses the legacy KNNModel which is not an Estimator subclass.
// If a KNNClassifier estimator wrapper is added later, this can be implemented.

// Generic: Apply parameters to any supported model type
static void apply_params(Estimator *model, ModelType type, const ParameterGrid *grid, const double *params) {
    switch (type) {
        case MODEL_LINEAR_REGRESSION:
            apply_linreg_params((LinearRegression*)model, grid, params);
            break;
        case MODEL_DECISION_TREE:
            apply_dt_params((DecisionTreeClassifier*)model, grid, params);
            break;
        case MODEL_NEURAL_NETWORK:
            apply_nn_params((NeuralNetwork*)model, grid, params);
            break;
        case MODEL_KNN:
            /* KNN uses legacy KNNModel, not tunable via GridSearch yet */
            break;
        default:
            /* Unknown model type — parameters silently ignored */
            break;
    }
}

int grid_search_cv_fit(GridSearchCV *gs, const Matrix *X, const Matrix *y) {
    if (!gs || !X || !y) return -1;

    int n_combos = gs->n_results;

    if (gs->verbose) {
        printf("Fitting %d folds for each of %d candidates, totalling %d fits\n",
               gs->cv, n_combos, gs->cv * n_combos);
    }

    clock_t total_start = clock();

    for (int combo = 0; combo < n_combos; combo++) {
        double *params = param_grid_get_combination(gs->param_grid, combo);
        if (!params) continue;

        // Clone and configure estimator
        Estimator *model = gs->base_estimator->clone(gs->base_estimator);
        if (!model) {
            free(params);
            continue;
        }

        // Apply parameters to any supported model type
        apply_params(model, gs->base_estimator->type, gs->param_grid, params);

        // Cross-validate
        clock_t fit_start = clock();
        CrossValResults *cv = cross_val_score(model, X, y, gs->cv, gs->shuffle, gs->seed);
        clock_t fit_end = clock();

        if (cv) {
            gs->results[combo].param_values = params;
            gs->results[combo].mean_test_score = cv->mean_test_score;
            gs->results[combo].std_test_score = cv->std_test_score;
            gs->results[combo].mean_train_score = cv->mean_train_score;
            gs->results[combo].mean_fit_time = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

            if (cv->mean_test_score > gs->best_score) {
                gs->best_score = cv->mean_test_score;
                gs->best_index = combo;
                free(gs->best_params);
                gs->best_params = malloc(gs->param_grid->n_params * sizeof(double));
                memcpy(gs->best_params, params, gs->param_grid->n_params * sizeof(double));
            }

            cross_val_results_free(cv);
        } else {
            free(params);
        }

        model->free(model);

        if (gs->verbose >= 2) {
            printf("[%d/%d] Score: %.6f\n", combo + 1, n_combos, gs->results[combo].mean_test_score);
        }
    }

    // Rank results
    for (int i = 0; i < n_combos; i++) {
        int rank = 1;
        for (int j = 0; j < n_combos; j++) {
            if (gs->results[j].mean_test_score > gs->results[i].mean_test_score) {
                rank++;
            }
        }
        gs->results[i].rank = rank;
    }

    // Refit best model on all data
    if (gs->refit && gs->best_index >= 0) {
        gs->best_estimator = gs->base_estimator->clone(gs->base_estimator);
        if (gs->best_estimator) {
            apply_params(gs->best_estimator, gs->base_estimator->type, gs->param_grid, gs->best_params);
        }
        if (gs->best_estimator) {
            gs->best_estimator->fit(gs->best_estimator, X, y);
        }
    }

    clock_t total_end = clock();

    if (gs->verbose) {
        printf("\nBest score: %.6f\n", gs->best_score);
        printf("Best parameters:\n");
        for (int i = 0; i < gs->param_grid->n_params; i++) {
            printf("  %s: %.6f\n", gs->param_grid->params[i].name, gs->best_params[i]);
        }
        printf("Total time: %.2f seconds\n", (double)(total_end - total_start) / CLOCKS_PER_SEC);
    }

    return 0;
}

const double* grid_search_cv_best_params(const GridSearchCV *gs) {
    return gs ? gs->best_params : NULL;
}

double grid_search_cv_best_score(const GridSearchCV *gs) {
    return gs ? gs->best_score : -1.0;
}

Estimator* grid_search_cv_best_estimator(GridSearchCV *gs) {
    return gs ? gs->best_estimator : NULL;
}

Matrix* grid_search_cv_predict(const GridSearchCV *gs, const Matrix *X) {
    if (!gs || !gs->best_estimator) return NULL;
    return gs->best_estimator->predict(gs->best_estimator, X);
}

double grid_search_cv_score(const GridSearchCV *gs, const Matrix *X, const Matrix *y) {
    if (!gs || !gs->best_estimator) return -1.0;
    return gs->best_estimator->score(gs->best_estimator, X, y);
}

void grid_search_cv_print_results(const GridSearchCV *gs) {
    if (!gs) return;

    printf("\n=== GridSearchCV Results ===\n\n");

    // Print header
    printf("%-6s ", "Rank");
    for (int i = 0; i < gs->param_grid->n_params; i++) {
        printf("%-15s ", gs->param_grid->params[i].name);
    }
    printf("%-15s %-15s\n", "Mean Score", "Std Score");
    printf("--------------------------------------------------------------\n");

    // Print results sorted by rank
    for (int rank = 1; rank <= gs->n_results; rank++) {
        for (int i = 0; i < gs->n_results; i++) {
            if (gs->results[i].rank == rank && gs->results[i].param_values) {
                printf("%-6d ", rank);
                for (int j = 0; j < gs->param_grid->n_params; j++) {
                    printf("%-15.6f ", gs->results[i].param_values[j]);
                }
                printf("%-15.6f %-15.6f\n",
                       gs->results[i].mean_test_score,
                       gs->results[i].std_test_score);
                break;  // Only print first with this rank
            }
        }
        if (rank >= 10) {
            printf("... (%d more)\n", gs->n_results - 10);
            break;
        }
    }

    printf("\nBest: rank=%d, score=%.6f\n", gs->results[gs->best_index].rank, gs->best_score);
    printf("============================\n\n");
}

void grid_search_cv_free(GridSearchCV *gs) {
    if (!gs) return;

    for (int i = 0; i < gs->n_results; i++) {
        free(gs->results[i].param_values);
    }
    free(gs->results);
    free(gs->best_params);

    if (gs->best_estimator) {
        gs->best_estimator->free(gs->best_estimator);
    }

    free(gs);
}

/* ============================================
 * Learning Curves Implementation
 * ============================================ */

LearningCurveResult* learning_curve(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    const double *train_sizes,
    int n_sizes,
    int cv
) {
    if (!estimator || !X || !y || !train_sizes || n_sizes <= 0) return NULL;

    LearningCurveResult *lc = calloc(1, sizeof(LearningCurveResult));
    if (!lc) return NULL;

    lc->n_points = n_sizes;
    lc->train_sizes = malloc(n_sizes * sizeof(size_t));
    lc->train_scores_mean = malloc(n_sizes * sizeof(double));
    lc->train_scores_std = malloc(n_sizes * sizeof(double));
    lc->test_scores_mean = malloc(n_sizes * sizeof(double));
    lc->test_scores_std = malloc(n_sizes * sizeof(double));

    if (!lc->train_sizes || !lc->train_scores_mean || !lc->train_scores_std ||
        !lc->test_scores_mean || !lc->test_scores_std) {
        learning_curve_free(lc);
        return NULL;
    }

    size_t n_samples = X->rows;

    for (int s = 0; s < n_sizes; s++) {
        // Convert fraction to absolute size
        size_t size;
        if (train_sizes[s] <= 1.0) {
            size = (size_t)(train_sizes[s] * n_samples);
        } else {
            size = (size_t)train_sizes[s];
        }
        if (size < 1) size = 1;
        if (size > n_samples) size = n_samples;
        lc->train_sizes[s] = size;

        // Create subset
        Matrix *X_sub = matrix_alloc(size, X->cols);
        Matrix *y_sub = matrix_alloc(size, y->cols);

        // Random sample
        size_t *indices = malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) indices[i] = i;
        shuffle_indices(indices, n_samples);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < X->cols; j++) {
                X_sub->data[i * X->cols + j] = X->data[indices[i] * X->cols + j];
            }
            for (size_t j = 0; j < y->cols; j++) {
                y_sub->data[i * y->cols + j] = y->data[indices[i] * y->cols + j];
            }
        }
        free(indices);

        // Cross-validate on subset
        CrossValResults *cv_result = cross_val_score(estimator, X_sub, y_sub, cv, 1, 42);

        if (cv_result) {
            lc->train_scores_mean[s] = cv_result->mean_train_score;
            lc->train_scores_std[s] = cv_result->std_train_score;
            lc->test_scores_mean[s] = cv_result->mean_test_score;
            lc->test_scores_std[s] = cv_result->std_test_score;
            cross_val_results_free(cv_result);
        }

        matrix_free(X_sub);
        matrix_free(y_sub);
    }

    return lc;
}

void learning_curve_print(const LearningCurveResult *lc) {
    if (!lc) return;

    printf("\n=== Learning Curve ===\n");
    printf("%-12s %-15s %-15s %-15s %-15s\n",
           "Train Size", "Train Mean", "Train Std", "Test Mean", "Test Std");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < lc->n_points; i++) {
        printf("%-12zu %-15.6f %-15.6f %-15.6f %-15.6f\n",
               lc->train_sizes[i],
               lc->train_scores_mean[i], lc->train_scores_std[i],
               lc->test_scores_mean[i], lc->test_scores_std[i]);
    }
    printf("======================\n\n");
}

int learning_curve_save_csv(const LearningCurveResult *lc, const char *filename) {
    if (!lc || !filename) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "train_size,train_score_mean,train_score_std,test_score_mean,test_score_std\n");
    for (int i = 0; i < lc->n_points; i++) {
        fprintf(f, "%zu,%.10f,%.10f,%.10f,%.10f\n",
                lc->train_sizes[i],
                lc->train_scores_mean[i], lc->train_scores_std[i],
                lc->test_scores_mean[i], lc->test_scores_std[i]);
    }

    fclose(f);
    return 0;
}

void learning_curve_free(LearningCurveResult *lc) {
    if (!lc) return;
    free(lc->train_sizes);
    free(lc->train_scores_mean);
    free(lc->train_scores_std);
    free(lc->test_scores_mean);
    free(lc->test_scores_std);
    free(lc);
}
