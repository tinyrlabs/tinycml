/**
 * decomposition.h - Dimensionality reduction algorithms
 *
 * Provides:
 * - PCA (Principal Component Analysis)
 * - Truncated SVD (for sparse data)
 */

#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include "matrix.h"
#include "estimator.h"

/**
 * PCA (Principal Component Analysis)
 *
 * Linear dimensionality reduction using Singular Value Decomposition
 */
typedef struct {
    Estimator base;

    // Configuration
    int n_components;           // Number of components to keep
    int whiten;                 // Whether to whiten components

    // Fitted attributes
    Matrix *components_;        // Principal axes (n_components x n_features)
    Matrix *mean_;              // Per-feature mean (1 x n_features)
    double *explained_variance_;         // Variance explained by each component
    double *explained_variance_ratio_;   // Percentage of variance explained
    double *singular_values_;            // Singular values
    int n_features_;
    int n_samples_;
    double total_variance_;
} PCA;

/* ============================================
 * PCA API
 * ============================================ */

/**
 * Create PCA transformer
 *
 * @param n_components Number of components to keep (0 = keep all)
 * @return PCA instance
 */
PCA* pca_create(int n_components);

/**
 * Create with whitening
 */
PCA* pca_create_full(int n_components, int whiten);

/**
 * Fit PCA to data
 */
Estimator* pca_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform data to reduced dimensionality
 */
Matrix* pca_transform(const Estimator *self, const Matrix *X);

/**
 * Fit and transform in one step
 */
Matrix* pca_fit_transform(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Inverse transform - reconstruct original data from reduced
 */
Matrix* pca_inverse_transform(const PCA *pca, const Matrix *X_reduced);

/**
 * Get explained variance ratio
 */
const double* pca_explained_variance_ratio(const PCA *pca);

/**
 * Get cumulative explained variance
 */
double pca_cumulative_variance(const PCA *pca, int n_components);

/**
 * Clone PCA
 */
Estimator* pca_clone(const Estimator *self);

/**
 * Free PCA
 */
void pca_free(Estimator *self);

/**
 * Print PCA summary
 */
void pca_print_summary(const Estimator *self);

/* ============================================
 * Helper functions
 * ============================================ */

/**
 * Compute covariance matrix
 */
Matrix* compute_covariance_matrix(const Matrix *X, const Matrix *mean);

/**
 * Power iteration for dominant eigenvector
 */
Matrix* power_iteration(const Matrix *A, int max_iter, double tol);

/**
 * Compute eigendecomposition (simplified - power iteration based)
 */
int eigen_decomposition(const Matrix *A, Matrix **eigenvectors, double *eigenvalues, int n_components);

#endif /* DECOMPOSITION_H */
