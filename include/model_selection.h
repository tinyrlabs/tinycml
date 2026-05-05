/**
 * model_selection.h - Model selection and hyperparameter tuning
 *
 * Provides scikit-learn style hyperparameter search:
 * - GridSearchCV: Exhaustive search over parameter grid
 * - RandomizedSearchCV: Random search over parameter distributions
 */

#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include "matrix.h"
#include "estimator.h"
#include "validation.h"

/**
 * Parameter types
 */
typedef enum {
    PARAM_INT,
    PARAM_DOUBLE,
    PARAM_ENUM
} ParamType;

/**
 * Single parameter specification
 */
typedef struct {
    const char *name;
    ParamType type;
    union {
        struct { int *values; int n_values; } int_vals;
        struct { double *values; int n_values; } double_vals;
        struct { int *values; int n_values; } enum_vals;
    } data;
} ParamSpec;

/**
 * Parameter grid
 */
typedef struct {
    ParamSpec *params;
    int n_params;
} ParameterGrid;

/**
 * Grid Search results for a single parameter combination
 */
typedef struct {
    double *param_values;      // Parameter values for this combination
    double mean_test_score;
    double std_test_score;
    double mean_train_score;
    double mean_fit_time;
    int rank;
} GridSearchResult;

/**
 * Grid Search CV structure
 */
typedef struct {
    Estimator *base_estimator;   // Template estimator to clone
    ParameterGrid *param_grid;
    int cv;                       // Number of cross-validation folds
    int shuffle;
    unsigned int seed;
    int verbose;
    int refit;                    // Whether to refit best model on all data

    // Results
    GridSearchResult *results;
    int n_results;
    int best_index;
    double best_score;
    Estimator *best_estimator;   // Refitted best estimator
    double *best_params;

    // Scoring
    int is_classification;
} GridSearchCV;

/* ============================================
 * Parameter Grid API
 * ============================================ */

/**
 * Create a parameter grid
 */
ParameterGrid* param_grid_create(int n_params);

/**
 * Add integer parameter
 */
int param_grid_add_int(ParameterGrid *grid, int index, const char *name,
                       const int *values, int n_values);

/**
 * Add double parameter
 */
int param_grid_add_double(ParameterGrid *grid, int index, const char *name,
                          const double *values, int n_values);

/**
 * Get total number of parameter combinations
 */
int param_grid_get_n_combinations(const ParameterGrid *grid);

/**
 * Get parameter combination by index
 */
double* param_grid_get_combination(const ParameterGrid *grid, int combo_index);

/**
 * Free parameter grid
 */
void param_grid_free(ParameterGrid *grid);

/* ============================================
 * GridSearchCV API
 * ============================================ */

/**
 * Create GridSearchCV
 *
 * @param estimator Base estimator to tune
 * @param param_grid Parameter grid to search
 * @param cv Number of cross-validation folds
 * @return GridSearchCV instance
 */
GridSearchCV* grid_search_cv_create(
    Estimator *estimator,
    ParameterGrid *param_grid,
    int cv
);

/**
 * Fit GridSearchCV - search all parameter combinations
 */
int grid_search_cv_fit(GridSearchCV *gs, const Matrix *X, const Matrix *y);

/**
 * Get best parameters
 */
const double* grid_search_cv_best_params(const GridSearchCV *gs);

/**
 * Get best score
 */
double grid_search_cv_best_score(const GridSearchCV *gs);

/**
 * Get best estimator (refitted on all data)
 */
Estimator* grid_search_cv_best_estimator(GridSearchCV *gs);

/**
 * Predict using best estimator
 */
Matrix* grid_search_cv_predict(const GridSearchCV *gs, const Matrix *X);

/**
 * Score using best estimator
 */
double grid_search_cv_score(const GridSearchCV *gs, const Matrix *X, const Matrix *y);

/**
 * Print CV results
 */
void grid_search_cv_print_results(const GridSearchCV *gs);

/**
 * Free GridSearchCV
 */
void grid_search_cv_free(GridSearchCV *gs);

/* ============================================
 * Learning Curves
 * ============================================ */

/**
 * Learning curve results
 */
typedef struct {
    size_t *train_sizes;
    double *train_scores_mean;
    double *train_scores_std;
    double *test_scores_mean;
    double *test_scores_std;
    int n_points;
} LearningCurveResult;

/**
 * Compute learning curve
 *
 * @param estimator Model to evaluate
 * @param X Feature matrix
 * @param y Target vector
 * @param train_sizes Array of training set sizes (as fractions 0-1 or absolute)
 * @param n_sizes Number of sizes to evaluate
 * @param cv Number of CV folds
 * @return Learning curve results
 */
LearningCurveResult* learning_curve(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    const double *train_sizes,
    int n_sizes,
    int cv
);

/**
 * Print learning curve
 */
void learning_curve_print(const LearningCurveResult *lc);

/**
 * Save learning curve to CSV
 */
int learning_curve_save_csv(const LearningCurveResult *lc, const char *filename);

/**
 * Free learning curve
 */
void learning_curve_free(LearningCurveResult *lc);

#endif /* MODEL_SELECTION_H */
