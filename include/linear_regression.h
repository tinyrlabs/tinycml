/**
 * @file linear_regression.h
 * @brief Linear regression for tinycml - Estimator API compatible
 */

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"
#include "estimator.h"

/**
 * Solver method for linear regression
 */
typedef enum {
    LINREG_SOLVER_CLOSED,    // Normal equation (X'X)^(-1)X'y
    LINREG_SOLVER_GD,        // Gradient descent
    LINREG_SOLVER_SGD,       // Stochastic gradient descent
    LINREG_SOLVER_AUTO       // Auto-select based on data size
} LinRegSolver;

/**
 * Linear Regression model structure
 */
typedef struct {
    Estimator base;           // Inherit from Estimator

    // Model parameters
    Matrix *weights;          // Learned weights (m x 1)
    Matrix *coef_;            // Coefficients without intercept (m-1 x 1)
    double intercept_;        // Intercept term

    // Hyperparameters
    LinRegSolver solver;
    double learning_rate;
    int max_iter;
    double tol;               // Tolerance for convergence
    int fit_intercept;        // Whether to fit intercept
    int normalize;            // Whether to normalize features

    // Training info
    int n_iter_;              // Actual iterations used
} LinearRegression;

/**
 * Create a new Linear Regression model
 *
 * @param solver Solver method (CLOSED, GD, SGD, AUTO)
 * @return New LinearRegression model, or NULL on error
 */
LinearRegression* linear_regression_create(LinRegSolver solver);

/**
 * Create with full configuration
 */
LinearRegression* linear_regression_create_full(
    LinRegSolver solver,
    double learning_rate,
    int max_iter,
    double tol,
    int fit_intercept,
    int normalize
);

/**
 * Fit the model (Estimator API)
 * Automatically adds bias column if fit_intercept is true
 */
Estimator* linear_regression_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict target values (Estimator API)
 */
Matrix* linear_regression_predict(const Estimator *self, const Matrix *X);

/**
 * Score using R² (Estimator API)
 */
double linear_regression_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone the model (Estimator API)
 */
Estimator* linear_regression_clone(const Estimator *self);

/**
 * Free the model (Estimator API)
 */
void linear_regression_free(Estimator *self);

/**
 * Save model to file
 */
int linear_regression_save(const Estimator *self, const char *filename);

/**
 * Load model from file
 */
Estimator* linear_regression_load(const char *filename);

/**
 * Print model summary
 */
void linear_regression_print_summary(const Estimator *self);

/**
 * Get coefficients (without intercept)
 */
const Matrix* linear_regression_get_coef(const LinearRegression *model);

/**
 * Get intercept
 */
double linear_regression_get_intercept(const LinearRegression *model);

/* ============================================
 * Legacy API (backward compatible)
 * ============================================ */

/**
 * @brief Fit linear regression using closed-form solution (normal equation)
 * @deprecated Use linear_regression_create() + fit() instead
 */
Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y);

/**
 * @brief Fit linear regression using gradient descent
 * @deprecated Use linear_regression_create_full() + fit() instead
 */
Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict target values
 * @deprecated Use linear_regression_predict() instead
 */
Matrix* linreg_predict(const Matrix *X, const Matrix *weights);

#endif /* LINEAR_REGRESSION_H */
