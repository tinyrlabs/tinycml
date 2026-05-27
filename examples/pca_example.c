/**
 * pca_example.c - Principal Component Analysis Demo
 *
 * Demonstrates:
 * - Dimensionality reduction with PCA
 * - Explained variance analysis
 * - Inverse transform for reconstruction
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "decomposition.h"
#include "utils.h"

// Generate correlated data
static void generate_correlated_data(Matrix **X, size_t n_samples, size_t n_features) {
    *X = matrix_alloc(n_samples, n_features);

    rand_seed(42);

    for (size_t i = 0; i < n_samples; i++) {
        // Base components (independent)
        double z1 = rand_normal_params(0, 3);
        double z2 = rand_normal();
        double z3 = rand_normal_params(0, 0.5);

        // Create correlated features from base components
        (*X)->data[i * n_features + 0] = z1 + 0.1 * z2;
        (*X)->data[i * n_features + 1] = 0.8 * z1 + 0.5 * z2;
        (*X)->data[i * n_features + 2] = 0.6 * z1 - 0.3 * z2 + z3;
        (*X)->data[i * n_features + 3] = 0.4 * z1 + 0.2 * z2 + 0.8 * z3;

        // Add small noise to remaining features
        for (size_t j = 4; j < n_features; j++) {
            (*X)->data[i * n_features + j] = rand_normal_params(0, 0.2);
        }
    }
}

int main(void) {
    printf("==============================================\n");
    printf("  tinycml - PCA Demo\n");
    printf("  Principal Component Analysis\n");
    printf("==============================================\n\n");

    // 1. Generate correlated data
    printf("1. Generating correlated data...\n");
    size_t n_samples = 100;
    size_t n_features = 6;
    Matrix *X;
    generate_correlated_data(&X, n_samples, n_features);
    printf("   Generated %zu samples with %zu features\n", n_samples, n_features);
    printf("   Features 0-3: correlated, Features 4-5: noise\n\n");

    // 2. Fit PCA with all components
    printf("2. Fitting PCA (all components)...\n");
    PCA *pca_full = pca_create(0);  // 0 = keep all
    pca_full->base.fit((Estimator*)pca_full, X, NULL);

    printf("   Explained variance ratio:\n");
    const double *evr = pca_explained_variance_ratio(pca_full);
    double cumulative = 0.0;
    for (int i = 0; i < pca_full->n_components; i++) {
        cumulative += evr[i];
        printf("   PC%d: %.4f (cumulative: %.4f)\n", i + 1, evr[i], cumulative);
    }
    printf("\n");

    // 3. Reduce to 2 components
    printf("3. Reducing to 2 components...\n");
    PCA *pca = pca_create(2);
    pca->base.fit((Estimator*)pca, X, NULL);

    Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);
    printf("   Original shape: %zu x %zu\n", X->rows, X->cols);
    printf("   Reduced shape:  %zu x %zu\n", X_reduced->rows, X_reduced->cols);
    printf("   Variance retained: %.2f%%\n\n",
           pca_cumulative_variance(pca, 2) * 100);

    // 4. Show some transformed samples
    printf("4. First 5 transformed samples:\n");
    printf("   Sample | PC1       | PC2\n");
    printf("   -------|-----------|----------\n");
    for (size_t i = 0; i < 5; i++) {
        printf("   %6zu | %9.4f | %9.4f\n",
               i,
               X_reduced->data[i * 2 + 0],
               X_reduced->data[i * 2 + 1]);
    }
    printf("\n");

    // 5. Inverse transform (reconstruction)
    printf("5. Reconstruction from 2 components...\n");
    Matrix *X_reconstructed = pca_inverse_transform(pca, X_reduced);

    if (X_reconstructed) {
        // Calculate reconstruction error (MSE)
        double mse = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            for (size_t j = 0; j < X->cols; j++) {
                double diff = X->data[i * X->cols + j] -
                              X_reconstructed->data[i * X_reconstructed->cols + j];
                mse += diff * diff;
            }
        }
        mse /= (X->rows * X->cols);

        printf("   Reconstruction MSE: %.6f\n", mse);
        printf("   (Lower is better - depends on variance discarded)\n\n");

        // Show original vs reconstructed for first sample
        printf("   Sample 0 - Original vs Reconstructed:\n");
        printf("   Feature | Original  | Reconstructed | Diff\n");
        printf("   --------|-----------|---------------|----------\n");
        for (size_t j = 0; j < X->cols; j++) {
            double orig = X->data[j];
            double recon = X_reconstructed->data[j];
            printf("   %7zu | %9.4f | %13.4f | %9.4f\n",
                   j, orig, recon, orig - recon);
        }
        printf("\n");

        matrix_free(X_reconstructed);
    }

    // 6. PCA with whitening
    printf("6. PCA with whitening...\n");
    PCA *pca_white = pca_create_full(2, 1);  // whiten=1
    pca_white->base.fit((Estimator*)pca_white, X, NULL);
    Matrix *X_whitened = pca_white->base.transform((Estimator*)pca_white, X);

    if (X_whitened) {
        // Check variance of whitened components (should be ~1)
        double var1 = 0.0, var2 = 0.0;
        double mean1 = 0.0, mean2 = 0.0;

        for (size_t i = 0; i < X_whitened->rows; i++) {
            mean1 += X_whitened->data[i * 2 + 0];
            mean2 += X_whitened->data[i * 2 + 1];
        }
        mean1 /= X_whitened->rows;
        mean2 /= X_whitened->rows;

        for (size_t i = 0; i < X_whitened->rows; i++) {
            double d1 = X_whitened->data[i * 2 + 0] - mean1;
            double d2 = X_whitened->data[i * 2 + 1] - mean2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        var1 /= X_whitened->rows;
        var2 /= X_whitened->rows;

        printf("   Whitened PC1 variance: %.4f (should be ~1.0)\n", var1);
        printf("   Whitened PC2 variance: %.4f (should be ~1.0)\n\n", var2);

        matrix_free(X_whitened);
    }

    // 7. Finding optimal number of components
    printf("7. Finding optimal n_components for 95%% variance...\n");
    double threshold = 0.95;
    cumulative = 0.0;
    int optimal_n = 0;
    for (int i = 0; i < pca_full->n_components; i++) {
        cumulative += evr[i];
        if (cumulative >= threshold) {
            optimal_n = i + 1;
            break;
        }
    }
    printf("   Need %d components for %.0f%% variance\n\n", optimal_n, threshold * 100);

    // 8. Print summary
    printf("8. Model Summary:\n");
    pca->base.print_summary((Estimator*)pca);

    // Cleanup
    printf("9. Cleaning up...\n");
    pca_full->base.free((Estimator*)pca_full);
    pca->base.free((Estimator*)pca);
    pca_white->base.free((Estimator*)pca_white);
    matrix_free(X);
    matrix_free(X_reduced);

    printf("   Done!\n\n");
    printf("==============================================\n");
    printf("  PCA demo completed!\n");
    printf("==============================================\n");

    return 0;
}
