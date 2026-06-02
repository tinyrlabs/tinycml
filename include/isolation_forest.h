/**
 * isolation_forest.h - Isolation Forest for anomaly detection
 *
 * Each tree isolates samples by randomly selecting a feature and split value.
 * Anomalies have shorter average path lengths.
 */

#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H

#include "matrix.h"
#include "estimator.h"

/**
 * Isolation Forest model
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    int n_trees;            /* Number of isolation trees (default: 100) */
    int max_samples;        /* Subsample size per tree (0 = min(256, n_samples)) */
    int max_depth;          /* Max tree depth (0 = ceil(log2(max_samples))) */
    unsigned int seed;      /* Random seed */

    /* Trained state */
    void **trees;           /* Array of IsolationTree pointers */
    int n_features;         /* Features seen during fit */
    size_t n_train;         /* Training set size */
    double *threshold;      /* Score threshold for anomaly (computed from training) */
} IsolationForest;

/**
 * Create an Isolation Forest
 *
 * @param n_trees      Number of isolation trees
 * @param max_samples  Subsample size per tree (0 = auto)
 * @return IsolationForest instance
 */
IsolationForest* isolation_forest_create(int n_trees, int max_samples);

/**
 * Fit the Isolation Forest
 */
Estimator* isolation_forest_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict: -1 for anomaly, 1 for normal
 */
Matrix* isolation_forest_predict(const Estimator *self, const Matrix *X);

/**
 * Score samples: anomaly score (higher = more anomalous)
 * Returns n_samples x 1 matrix of scores
 */
Matrix* isolation_forest_score_samples(const Estimator *self, const Matrix *X);

/**
 * Score: fraction of correct predictions if y contains -1/1 labels
 */
double isolation_forest_score(const Estimator *self, const Matrix *X, const Matrix *y);

Estimator* isolation_forest_clone(const Estimator *self);
void isolation_forest_free(Estimator *self);

#endif /* ISOLATION_FOREST_H */
