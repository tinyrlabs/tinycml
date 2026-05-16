/**
 * @file test_dbscan.c
 * @brief Unit tests for DBSCAN clustering
 */

#include "test_harness.h"
#include "matrix.h"
#include "dbscan.h"

TEST(test_dbscan_create_free) {
    DBSCAN *db = dbscan_create();
    ASSERT_NOT_NULL(db);
    ASSERT_NEAR(db->epsilon, 0.5, 1e-9);
    ASSERT_EQ(db->min_samples, 5);

    dbscan_free(db);
}

TEST(test_dbscan_two_blobs) {
    /* Two clear clusters + some noise points */
    int N = 30;
    Matrix *X = matrix_alloc(N, 2);
    ASSERT_NOT_NULL(X);

    /* Cluster 1: centered around (0, 0) */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, i, 0, 0.0 + (i % 5) * 0.1);
        matrix_set(X, i, 1, 0.0 + (i / 5) * 0.1);
    }

    /* Cluster 2: centered around (10, 10) */
    for (int i = 0; i < 10; i++) {
        matrix_set(X, 10 + i, 0, 10.0 + (i % 5) * 0.1);
        matrix_set(X, 10 + i, 1, 10.0 + (i / 5) * 0.1);
    }

    /* Noise points: far from both clusters */
    matrix_set(X, 20, 0, -10.0);
    matrix_set(X, 20, 1, -10.0);
    matrix_set(X, 21, 0, 20.0);
    matrix_set(X, 21, 1, 20.0);
    matrix_set(X, 22, 0, -20.0);
    matrix_set(X, 22, 1, 30.0);

    /* More points in cluster 1 to reach min_samples */
    matrix_set(X, 23, 0, 0.5);
    matrix_set(X, 23, 1, 0.5);
    matrix_set(X, 24, 0, -0.3);
    matrix_set(X, 24, 1, 0.2);
    /* More points in cluster 2 */
    matrix_set(X, 25, 0, 10.5);
    matrix_set(X, 25, 1, 10.5);
    matrix_set(X, 26, 0, 9.8);
    matrix_set(X, 26, 1, 10.2);
    /* More noise */
    matrix_set(X, 27, 0, -15.0);
    matrix_set(X, 27, 1, -15.0);
    matrix_set(X, 28, 0, 25.0);
    matrix_set(X, 28, 1, 25.0);
    matrix_set(X, 29, 0, 50.0);
    matrix_set(X, 29, 1, 50.0);

    DBSCAN *db = dbscan_create();
    ASSERT_NOT_NULL(db);
    db->epsilon = 2.0;
    db->min_samples = 3;

    int *labels = dbscan_fit(db, X);
    ASSERT_NOT_NULL(labels);

    int n_clusters = dbscan_count_clusters(labels, N);
    ASSERT_EQ(n_clusters, 2);

    /* Noise points should have label -1 */
    int noise_count = 0;
    for (int i = 20; i < 23; i++) {
        if (labels[i] == -1) noise_count++;
    }
    for (int i = 27; i < 30; i++) {
        if (labels[i] == -1) noise_count++;
    }
    ASSERT(noise_count >= 3);

    free(labels);
    matrix_free(X);
    dbscan_free(db);
}

TEST(test_dbscan_noise) {
    /* Sparse outliers — all should be noise */
    int N = 10;
    Matrix *X = matrix_alloc(N, 2);
    ASSERT_NOT_NULL(X);

    /* Points far apart from each other */
    for (int i = 0; i < N; i++) {
        matrix_set(X, i, 0, (double)(i * 10));
        matrix_set(X, i, 1, (double)(i * 10));
    }

    DBSCAN *db = dbscan_create();
    ASSERT_NOT_NULL(db);
    db->epsilon = 0.5;
    db->min_samples = 3;

    int *labels = dbscan_fit(db, X);
    ASSERT_NOT_NULL(labels);

    /* With high min_samples and low epsilon, most/all should be noise */
    int noise_count = 0;
    for (int i = 0; i < N; i++) {
        if (labels[i] == -1) noise_count++;
    }
    ASSERT(noise_count > 0);

    free(labels);
    matrix_free(X);
    dbscan_free(db);
}

TEST(test_dbscan_single_cluster) {
    /* One tight cluster, all points should have same label */
    int N = 15;
    Matrix *X = matrix_alloc(N, 2);
    ASSERT_NOT_NULL(X);

    /* All points close together */
    for (int i = 0; i < N; i++) {
        matrix_set(X, i, 0, (double)(i % 5) * 0.1);
        matrix_set(X, i, 1, (double)(i / 5) * 0.1);
    }

    DBSCAN *db = dbscan_create();
    ASSERT_NOT_NULL(db);
    db->epsilon = 1.0;
    db->min_samples = 3;

    int *labels = dbscan_fit(db, X);
    ASSERT_NOT_NULL(labels);

    int n_clusters = dbscan_count_clusters(labels, N);
    ASSERT_EQ(n_clusters, 1);

    /* All points should be in cluster 0 (not noise) */
    for (int i = 0; i < N; i++) {
        ASSERT_EQ(labels[i], 0);
    }

    free(labels);
    matrix_free(X);
    dbscan_free(db);
}

int main(void) {
    RUN_TEST(test_dbscan_create_free);
    RUN_TEST(test_dbscan_two_blobs);
    RUN_TEST(test_dbscan_noise);
    RUN_TEST(test_dbscan_single_cluster);
    TEST_SUMMARY();
}
