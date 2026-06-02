/**
 * sgd.h - Stochastic Gradient Descent classifier and regressor
 *
 * Linear models trained with SGD: supports log-loss (classification)
 * and squared loss (regression). Scikit-learn compatible API.
 */

#ifndef SGD_H
#define SGD_H

#include "matrix.h"
#include "estimator.h"

/**
 * Loss function type
 */
typedef enum {
    SGD_LOSS_SQUARED,       /* MSE for regression */
    SGD_LOSS_LOG,           /* Log-loss for classification */
    SGD_LOSS_HINGE,         /* SVM-style hinge loss */
    SGD_LOSS_HUBER          /* Huber loss (robust regression) */
} SGDLossType;

/**
 * SGD Classifier / Regressor model
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    SGDLossType loss;
    double alpha;           /* L2 regularization strength */
    double learning_rate;   /* Initial learning rate */
    double eta0;            /* Constant learning rate (when lr=constant) */
    int max_iter;           /* Maximum epochs */
    double tol;             /* Early stopping tolerance (0 = disabled) */
    int shuffle;            /* Shuffle data each epoch */
    unsigned int seed;      /* Random seed */

    /* Trained state */
    double *weights;        /* Weight vector (n_features) */
    double bias;            /* Intercept */
    int n_features;         /* Number of features seen during fit */
    int n_classes;          /* Number of classes (1=regression) */
    int converged;          /* Whether early stopping triggered */
} SGDModel;

/**
 * Create an SGD model
 *
 * @param loss     Loss function
 * @param alpha    L2 regularization (default 0.0001)
 * @param lr       Learning rate (default 0.01)
 * @param max_iter Maximum epochs (default 1000)
 * @return SGDModel instance
 */
SGDModel* sgd_create(SGDLossType loss, double alpha, double lr, int max_iter);

/**
 * Convenience: create SGD classifier with log loss
 */
SGDModel* sgd_classifier_create(double alpha, double lr, int max_iter);

/**
 * Convenience: create SGD regressor with squared loss
 */
SGDModel* sgd_regressor_create(double alpha, double lr, int max_iter);

/* Estimator vtable functions */
Estimator* sgd_fit(Estimator *self, const Matrix *X, const Matrix *y);
Matrix* sgd_predict(const Estimator *self, const Matrix *X);
Matrix* sgd_predict_proba(const Estimator *self, const Matrix *X);
double sgd_score(const Estimator *self, const Matrix *X, const Matrix *y);
Estimator* sgd_clone(const Estimator *self);
void sgd_free(Estimator *self);
int sgd_save(const Estimator *self, const char *path);
Estimator* sgd_load(const char *path);

#endif /* SGD_H */
