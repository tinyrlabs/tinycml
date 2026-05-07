/**
 * decomposition.c - PCA implementation using power iteration
 *
 * Uses iterative method for eigenvector computation - pure C, no LAPACK dependency
 */

#include "decomposition.h"
#include "vector.h"
#include "utils.h"
#include "cml_error.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Helper Functions
 * ============================================ */

Matrix* compute_covariance_matrix(const Matrix *X, const Matrix *mean) {
    size_t n = X->rows;
    size_t m = X->cols;

    // Center data
    Matrix *X_centered = matrix_alloc(n, m);
    if (!X_centered) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            X_centered->data[i * m + j] = X->data[i * m + j] - mean->data[j];
        }
    }

    // Cov = X_centered' @ X_centered / (n - 1)
    Matrix *X_t = matrix_transpose(X_centered);
    Matrix *cov = matrix_matmul(X_t, X_centered);

    if (cov) {
        double scale = (n > 1) ? 1.0 / (n - 1) : 1.0;
        for (size_t i = 0; i < cov->rows * cov->cols; i++) {
            cov->data[i] *= scale;
        }
    }

    matrix_free(X_centered);
    matrix_free(X_t);

    return cov;
}

Matrix* power_iteration(const Matrix *A, int max_iter, double tol) {
    size_t n = A->rows;

    // Initialize random vector
    Matrix *v = matrix_alloc(n, 1);
    if (!v) return NULL;

    for (size_t i = 0; i < n; i++) {
        v->data[i] = rand_uniform() - 0.5;
    }

    // Normalize
    double norm = vector_norm(v);
    for (size_t i = 0; i < n; i++) {
        v->data[i] /= norm;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // v_new = A @ v
        Matrix *v_new = matrix_matmul(A, v);
        if (!v_new) {
            matrix_free(v);
            return NULL;
        }

        // Normalize
        norm = vector_norm(v_new);
        if (norm < 1e-10) {
            matrix_free(v_new);
            break;
        }
        for (size_t i = 0; i < n; i++) {
            v_new->data[i] /= norm;
        }

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = fabs(v_new->data[i]) - fabs(v->data[i]);
            diff += d * d;
        }
        diff = sqrt(diff);

        matrix_free(v);
        v = v_new;

        if (diff < tol) break;
    }

    return v;
}

// Deflation: A = A - eigenvalue * v @ v.T
static void deflate_matrix(Matrix *A, const Matrix *v, double eigenvalue) {
    size_t n = A->rows;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A->data[i * n + j] -= eigenvalue * v->data[i] * v->data[j];
        }
    }
}

int eigen_decomposition(const Matrix *A, Matrix **eigenvectors, double *eigenvalues, int n_components) {
    size_t n = A->rows;

    // Create copy of A for deflation
    Matrix *A_work = matrix_copy(A);
    if (!A_work) return -1;

    *eigenvectors = matrix_alloc(n_components, n);
    if (!*eigenvectors) {
        matrix_free(A_work);
        return -1;
    }

    for (int k = 0; k < n_components; k++) {
        // Find dominant eigenvector
        Matrix *v = power_iteration(A_work, 1000, 1e-8);
        if (!v) {
            matrix_free(A_work);
            matrix_free(*eigenvectors);
            return -1;
        }

        // Compute eigenvalue: lambda = v.T @ A @ v
        Matrix *Av = matrix_matmul(A_work, v);
        double eigenvalue = 0.0;
        for (size_t i = 0; i < n; i++) {
            eigenvalue += v->data[i] * Av->data[i];
        }
        matrix_free(Av);

        eigenvalues[k] = eigenvalue;

        // Store eigenvector as row
        for (size_t i = 0; i < n; i++) {
            (*eigenvectors)->data[k * n + i] = v->data[i];
        }

        // Deflate matrix
        deflate_matrix(A_work, v, eigenvalue);

        matrix_free(v);
    }

    matrix_free(A_work);
    return 0;
}

/* ============================================
 * PCA Implementation
 * ============================================ */

PCA* pca_create(int n_components) {
    return pca_create_full(n_components, 0);
}

PCA* pca_create_full(int n_components, int whiten) {
    PCA *pca = calloc(1, sizeof(PCA));
    if (!pca) return NULL;

    pca->base.type = MODEL_PCA;
    pca->base.task = TASK_TRANSFORMATION;
    pca->base.is_fitted = 0;
    pca->base.verbose = VERBOSE_SILENT;

    pca->base.fit = pca_fit;
    pca->base.predict = NULL;
    pca->base.predict_proba = NULL;
    pca->base.transform = pca_transform;
    pca->base.score = NULL;
    pca->base.clone = pca_clone;
    pca->base.free = pca_free;
    pca->base.save = NULL;
    pca->base.load = NULL;
    pca->base.print_summary = pca_print_summary;

    pca->n_components = n_components;
    pca->whiten = whiten;
    pca->components_ = NULL;
    pca->mean_ = NULL;
    pca->explained_variance_ = NULL;
    pca->explained_variance_ratio_ = NULL;
    pca->singular_values_ = NULL;

    return pca;
}

Estimator* pca_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;  // PCA doesn't use y
    PCA *pca = (PCA*)self;

    pca->n_samples_ = X->rows;
    pca->n_features_ = X->cols;

    // Determine number of components
    int n_comp = pca->n_components;
    if (n_comp <= 0 || n_comp > (int)X->cols) {
        n_comp = X->cols;
    }
    pca->n_components = n_comp;

    // Compute mean
    pca->mean_ = matrix_alloc(1, X->cols);
    for (size_t j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            sum += X->data[i * X->cols + j];
        }
        pca->mean_->data[j] = sum / X->rows;
    }

    // Compute covariance matrix
    Matrix *cov = compute_covariance_matrix(X, pca->mean_);
    if (!cov) return NULL;

    // Compute total variance
    pca->total_variance_ = 0.0;
    for (size_t i = 0; i < cov->rows; i++) {
        pca->total_variance_ += cov->data[i * cov->cols + i];
    }

    // Eigendecomposition
    pca->explained_variance_ = malloc(n_comp * sizeof(double));
    if (eigen_decomposition(cov, &pca->components_, pca->explained_variance_, n_comp) != 0) {
        matrix_free(cov);
        return NULL;
    }

    matrix_free(cov);

    // Compute explained variance ratio
    pca->explained_variance_ratio_ = malloc(n_comp * sizeof(double));
    pca->singular_values_ = malloc(n_comp * sizeof(double));

    for (int k = 0; k < n_comp; k++) {
        pca->explained_variance_ratio_[k] = pca->explained_variance_[k] / pca->total_variance_;
        pca->singular_values_[k] = sqrt(pca->explained_variance_[k] * (X->rows - 1));
    }

    pca->base.is_fitted = 1;

    if (pca->base.verbose >= VERBOSE_MINIMAL) {
        printf("PCA fitted: %d components, %.2f%% variance explained\n",
               n_comp, pca_cumulative_variance(pca, n_comp) * 100);
    }

    return self;
}

Matrix* pca_transform(const Estimator *self, const Matrix *X) {
    const PCA *pca = (const PCA*)self;

    if (!pca->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "PCA not fitted");
        return NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;
    int k = pca->n_components;

    // Center data
    Matrix *X_centered = matrix_alloc(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            X_centered->data[i * m + j] = X->data[i * m + j] - pca->mean_->data[j];
        }
    }

    // Project: X_reduced = X_centered @ components.T
    Matrix *components_t = matrix_transpose(pca->components_);
    Matrix *X_reduced = matrix_matmul(X_centered, components_t);

    matrix_free(X_centered);
    matrix_free(components_t);

    // Whiten if requested
    if (pca->whiten && X_reduced) {
        for (size_t i = 0; i < X_reduced->rows; i++) {
            for (int j = 0; j < k; j++) {
                if (pca->singular_values_[j] > 1e-10) {
                    X_reduced->data[i * k + j] /= pca->singular_values_[j];
                }
            }
        }
    }

    return X_reduced;
}

Matrix* pca_fit_transform(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!pca_fit(self, X, y)) return NULL;
    return pca_transform(self, X);
}

Matrix* pca_inverse_transform(const PCA *pca, const Matrix *X_reduced) {
    if (!pca->base.is_fitted) return NULL;

    // X_original = X_reduced @ components + mean
    Matrix *X_reconstructed = matrix_matmul(X_reduced, pca->components_);
    if (!X_reconstructed) return NULL;

    // Add mean back
    for (size_t i = 0; i < X_reconstructed->rows; i++) {
        for (size_t j = 0; j < X_reconstructed->cols; j++) {
            X_reconstructed->data[i * X_reconstructed->cols + j] += pca->mean_->data[j];
        }
    }

    return X_reconstructed;
}

const double* pca_explained_variance_ratio(const PCA *pca) {
    return pca ? pca->explained_variance_ratio_ : NULL;
}

double pca_cumulative_variance(const PCA *pca, int n_components) {
    if (!pca || !pca->explained_variance_ratio_) return 0.0;

    double sum = 0.0;
    int n = n_components;
    if (n > pca->n_components) n = pca->n_components;

    for (int i = 0; i < n; i++) {
        sum += pca->explained_variance_ratio_[i];
    }
    return sum;
}

Estimator* pca_clone(const Estimator *self) {
    const PCA *pca = (const PCA*)self;
    return (Estimator*)pca_create_full(pca->n_components, pca->whiten);
}

void pca_free(Estimator *self) {
    PCA *pca = (PCA*)self;
    if (!pca) return;

    matrix_free(pca->components_);
    matrix_free(pca->mean_);
    free(pca->explained_variance_);
    free(pca->explained_variance_ratio_);
    free(pca->singular_values_);
    free(pca);
}

void pca_print_summary(const Estimator *self) {
    const PCA *pca = (const PCA*)self;

    printf("\n=== PCA Summary ===\n");
    printf("Fitted: %s\n", pca->base.is_fitted ? "Yes" : "No");
    printf("Components: %d\n", pca->n_components);
    printf("Whiten: %s\n", pca->whiten ? "Yes" : "No");

    if (pca->base.is_fitted) {
        printf("\nExplained variance ratio:\n");
        double cumulative = 0.0;
        for (int i = 0; i < pca->n_components && i < 10; i++) {
            cumulative += pca->explained_variance_ratio_[i];
            printf("  PC%d: %.4f (cumulative: %.4f)\n",
                   i + 1, pca->explained_variance_ratio_[i], cumulative);
        }
        if (pca->n_components > 10) {
            printf("  ... (%d more)\n", pca->n_components - 10);
        }
        printf("\nTotal variance explained: %.4f\n", pca_cumulative_variance(pca, pca->n_components));
    }
    printf("===================\n\n");
}
