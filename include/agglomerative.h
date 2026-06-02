/**
 * agglomerative.h - Agglomerative (hierarchical) clustering
 *
 * Bottom-up clustering with single, complete, and average linkage.
 */

#ifndef AGGLOMERATIVE_H
#define AGGLOMERATIVE_H

#include "matrix.h"
#include "estimator.h"

/**
 * Linkage criterion
 */
typedef enum {
    LINKAGE_SINGLE,     /* Minimum distance between clusters */
    LINKAGE_COMPLETE,   /* Maximum distance between clusters */
    LINKAGE_AVERAGE     /* Average distance between clusters */
} LinkageType;

/**
 * Agglomerative Clustering model
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    int n_clusters;     /* Target number of clusters */
    LinkageType linkage;

    /* Trained state */
    int *labels;        /* Cluster assignments for training data */
    size_t n_samples;   /* Number of training samples */
    int n_features;     /* Feature count */
} AgglomerativeClustering;

/**
 * Create an Agglomerative Clustering model
 *
 * @param n_clusters  Number of target clusters
 * @param linkage     Linkage criterion
 * @return AgglomerativeClustering instance
 */
AgglomerativeClustering* agglomerative_create(int n_clusters, LinkageType linkage);

/**
 * Fit: compute hierarchical clustering
 */
Estimator* agglomerative_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict: assign new points to nearest cluster centroid
 */
Matrix* agglomerative_predict(const Estimator *self, const Matrix *X);

/* Score: silhouette score on training data */
double agglomerative_score(const Estimator *self, const Matrix *X, const Matrix *y);

Estimator* agglomerative_clone(const Estimator *self);
void agglomerative_free(Estimator *self);

#endif /* AGGLOMERATIVE_H */
