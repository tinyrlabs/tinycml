/**
 * @file ridge.h
 * @brief Ridge (L2-regularized) regression for tinycml
 */

#ifndef CML_RIDGE_H
#define CML_RIDGE_H

#include "matrix.h"
#include "estimator.h"

typedef struct {
    Estimator base;
    double alpha;      /* L2 penalty (default 1.0) */
    int max_iter;      /* unused (closed-form), kept for API compat */
    double tol;        /* unused (closed-form), kept for API compat */
    Matrix *weights;   /* fitted weights (n_features x 1) */
    double bias;       /* fitted intercept */
    int n_features;    /* number of features seen in fit */
} RidgeModel;

/**
 * @brief Create a new Ridge model with default alpha=1.0
 */
RidgeModel* ridge_model_create(void);

/**
 * @brief Estimator vtable: fit
 */
Estimator* ridge_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: predict
 */
Matrix* ridge_predict(const Estimator *self, const Matrix *X);

/**
 * @brief Estimator vtable: score (R²)
 */
double ridge_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: clone
 */
Estimator* ridge_clone(const Estimator *self);

/**
 * @brief Estimator vtable: free
 */
void ridge_free(Estimator *self);

#endif /* CML_RIDGE_H */
