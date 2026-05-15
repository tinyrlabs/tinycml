/**
 * @file lasso.h
 * @brief Lasso (L1-regularized) regression for tinycml
 */

#ifndef CML_LASSO_H
#define CML_LASSO_H

#include "matrix.h"
#include "estimator.h"

typedef struct {
    Estimator base;
    double alpha;      /* L1 penalty (default 1.0) */
    int max_iter;      /* max coordinate-descent iterations */
    double tol;        /* convergence tolerance */
    Matrix *weights;   /* fitted weights (n_features x 1) */
    double bias;       /* fitted intercept */
    int n_features;    /* number of features seen in fit */
} LassoModel;

/**
 * @brief Create a new Lasso model with default alpha=1.0, max_iter=1000, tol=1e-6
 */
LassoModel* lasso_model_create(void);

/**
 * @brief Estimator vtable: fit
 */
Estimator* lasso_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: predict
 */
Matrix* lasso_predict(const Estimator *self, const Matrix *X);

/**
 * @brief Estimator vtable: score (R²)
 */
double lasso_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: clone
 */
Estimator* lasso_clone(const Estimator *self);

/**
 * @brief Estimator vtable: free
 */
void lasso_free(Estimator *self);

#endif /* CML_LASSO_H */
