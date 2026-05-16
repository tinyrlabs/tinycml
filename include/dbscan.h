/**
 * @file dbscan.h
 * @brief DBSCAN density-based clustering
 */

#ifndef CML_DBSCAN_H
#define CML_DBSCAN_H

#include "matrix.h"

/**
 * @brief DBSCAN clustering model
 */
typedef struct {
    double epsilon;     /**< Neighborhood radius (default 0.5) */
    int min_samples;    /**< Minimum neighbors (default 5) */
} DBSCAN;

/**
 * @brief Fit DBSCAN model to data
 * @param model DBSCAN model with epsilon and min_samples set
 * @param X Data matrix (n_samples x n_features)
 * @return Array of cluster labels (n_samples). -1 = noise. Caller must free.
 */
int* dbscan_fit(const DBSCAN *model, const Matrix *X);

/**
 * @brief Count number of distinct clusters in labels
 * @param labels Cluster label array
 * @param n Number of labels
 * @return Number of clusters (ignoring noise label -1)
 */
int dbscan_count_clusters(const int *labels, int n);

/**
 * @brief Create a DBSCAN model with default parameters
 * @return Newly allocated DBSCAN model, or NULL on failure
 */
DBSCAN* dbscan_create(void);

/**
 * @brief Free a DBSCAN model
 * @param model Model to free
 */
void dbscan_free(DBSCAN *model);

#endif /* CML_DBSCAN_H */
