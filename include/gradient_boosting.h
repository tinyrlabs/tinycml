/**
 * gradient_boosting.h - Gradient Boosted Decision Trees (GBDT)
 *
 * Implements gradient boosting for:
 * - Regression (MSE loss)
 * - Binary classification (log-loss / cross-entropy)
 * - Multi-class classification (softmax / one-vs-rest)
 *
 * Uses DecisionTree as weak learner (max_depth=3 default).
 * Compatible with the Estimator vtable API.
 */

#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "matrix.h"
#include "estimator.h"
#include "decision_tree.h"

/**
 * Gradient Boosting model
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    int n_estimators;           /* Number of boosting rounds (default: 100) */
    double learning_rate;       /* Shrinkage factor (default: 0.1) */
    int max_depth;              /* Max depth of each weak learner (default: 3) */
    int min_samples_split;      /* Min samples to split (default: 2) */
    double subsample;           /* Fraction of samples for stochastic GB (default: 1.0) */
    unsigned int seed;          /* Random seed for subsampling */

    /* Trained state */
    DecisionTreeRegressor **trees;  /* Array of fitted trees */
    double init_prediction;         /* Initial prediction (mean / log-odds) */
    int n_classes;                  /* Number of classes (1 for regression, 2+ for classification) */
    int n_features;                 /* Number of features seen during fit */
} GradientBoosting;

/* ============================================
 * GradientBoosting API
 * ============================================ */

/**
 * Create a GradientBoosting model
 *
 * @param n_estimators   Number of boosting rounds
 * @param learning_rate  Shrinkage / step size
 * @param max_depth      Max depth of each weak learner tree
 * @param min_samples_split  Minimum samples to split
 * @param subsample      Fraction of training samples per round (1.0 = full)
 * @return GradientBoosting instance (wrapped as Estimator*)
 */
GradientBoosting* gradient_boosting_create(
    int n_estimators,
    double learning_rate,
    int max_depth,
    int min_samples_split,
    double subsample
);

/**
 * Fit (vtable entry)
 */
Estimator* gradient_boosting_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict (vtable entry)
 */
Matrix* gradient_boosting_predict(const Estimator *self, const Matrix *X);

/**
 * Predict probabilities (vtable entry)
 */
Matrix* gradient_boosting_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score (vtable entry): R² for regression, accuracy for classification
 */
double gradient_boosting_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone (vtable entry)
 */
Estimator* gradient_boosting_clone(const Estimator *self);

/**
 * Free (vtable entry)
 */
void gradient_boosting_free(Estimator *self);

/**
 * Save model to binary file
 *
 * @param self  Estimator pointer
 * @param path  File path
 * @return 0 on success, -1 on error
 */
int gradient_boosting_save(const Estimator *self, const char *path);

/**
 * Load model from binary file
 *
 * @param path  File path
 * @return Estimator pointer on success, NULL on error
 */
Estimator* gradient_boosting_load(const char *path);

#endif /* GRADIENT_BOOSTING_H */
