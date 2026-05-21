/**
 * @file test_silhouette.c
 * @brief Tests for Silhouette Score clustering metric
 */

#include "test_harness.h"
#include "metrics.h"
#include "matrix.h"

TEST(test_silhouette_well_separated) {
    /* Two clusters: cluster 0 centered at (0,0), cluster 1 at (5,5) */
    Matrix *X = matrix_alloc(6, 2);
    /* Cluster 0 */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 0.5); matrix_set(X, 1, 1, 0.5);
    matrix_set(X, 2, 0, -0.5); matrix_set(X, 2, 1, -0.5);
    /* Cluster 1 */
    matrix_set(X, 3, 0, 5.0); matrix_set(X, 3, 1, 5.0);
    matrix_set(X, 4, 0, 5.5); matrix_set(X, 4, 1, 5.5);
    matrix_set(X, 5, 0, 4.5); matrix_set(X, 5, 1, 4.5);

    Matrix *labels = matrix_alloc(6, 1);
    labels->data[0] = 0.0;
    labels->data[1] = 0.0;
    labels->data[2] = 0.0;
    labels->data[3] = 1.0;
    labels->data[4] = 1.0;
    labels->data[5] = 1.0;

    double score = silhouette_score(X, labels, 2);
    /* Well-separated clusters should have high silhouette (> 0.7) */
    ASSERT(score > 0.7);
    ASSERT(score <= 1.0);

    matrix_free(X);
    matrix_free(labels);
}

TEST(test_silhouette_null_inputs) {
    Matrix *X = matrix_alloc(4, 2);
    Matrix *labels = matrix_alloc(4, 1);
    matrix_set(labels, 0, 0, 0.0);
    matrix_set(labels, 1, 0, 0.0);
    matrix_set(labels, 2, 0, 1.0);
    matrix_set(labels, 3, 0, 1.0);

    ASSERT(silhouette_score(NULL, labels, 2) == 0.0);
    ASSERT(silhouette_score(X, NULL, 2) == 0.0);
    ASSERT(silhouette_score(X, labels, 1) == 0.0);
    ASSERT(silhouette_score(X, labels, 0) == 0.0);

    matrix_free(X);
    matrix_free(labels);
}

TEST(test_silhouette_single_cluster_returns_zero) {
    Matrix *X = matrix_alloc(4, 2);
    Matrix *labels = matrix_alloc(4, 1);
    for (int i = 0; i < 4; i++) {
        matrix_set(X, i, 0, (double)i);
        matrix_set(X, i, 1, (double)i);
        labels->data[i] = 0.0;
    }
    /* n_clusters=1 returns 0.0 (silhouette undefined for single cluster) */
    ASSERT(silhouette_score(X, labels, 1) == 0.0);

    matrix_free(X);
    matrix_free(labels);
}

TEST(test_silhouette_range) {
    /* Two identical clusters — should give low/zero silhouette */
    Matrix *X = matrix_alloc(4, 2);
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 1.0);
    matrix_set(X, 2, 0, 0.5); matrix_set(X, 2, 1, 0.5);
    matrix_set(X, 3, 0, 1.5); matrix_set(X, 3, 1, 1.5);

    Matrix *labels = matrix_alloc(4, 1);
    labels->data[0] = 0.0;
    labels->data[1] = 0.0;
    labels->data[2] = 1.0;
    labels->data[3] = 1.0;

    double score = silhouette_score(X, labels, 2);
    /* Score should be in valid range [-1, 1] */
    ASSERT(score >= -1.0);
    ASSERT(score <= 1.0);

    matrix_free(X);
    matrix_free(labels);
}

int main(void) {
    printf("Silhouette Score Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_silhouette_well_separated);
    RUN_TEST(test_silhouette_null_inputs);
    RUN_TEST(test_silhouette_single_cluster_returns_zero);
    RUN_TEST(test_silhouette_range);

    TEST_SUMMARY();
}
