/**
 * @file kmeans.h
 * @brief k-Means clustering
 */

#ifndef KMEANS_H
#define KMEANS_H

#include "matrix.h"

/**
 * @brief k-Means model structure
 */
typedef struct {
    Matrix *centroids;  /**< Cluster centroids (k x n_features) */
    int k;              /**< Number of clusters */
    int max_iter;       /**< Maximum iterations */
} KMeansModel;

/**
 * @brief Fit k-Means model using Lloyd's algorithm
 * @param X Data matrix (n_samples x n_features)
 * @param k Number of clusters
 * @param max_iter Maximum iterations
 * @param seed Random seed for initialization
 * @return Fitted model, or NULL on error
 */
KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed);

/**
 * @brief Predict cluster labels for samples
 * @param model Fitted k-Means model
 * @param X Data matrix
 * @return Cluster labels (n_samples x 1), or NULL on error
 */
Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X);

/**
 * @brief Free k-Means model
 * @param model Model to free
 */
void kmeans_free(KMeansModel *model);

#endif /* KMEANS_H */
