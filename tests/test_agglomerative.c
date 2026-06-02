/**
 * test_agglomerative.c - Tests for Agglomerative Clustering
 */

#include "test_harness.h"
#include "agglomerative.h"
#include "matrix.h"

TEST(test_agg_create) {
    AgglomerativeClustering *ac = agglomerative_create(2, LINKAGE_AVERAGE);
    ASSERT_NOT_NULL(ac);
    ASSERT_EQ(ac->n_clusters, 2);
    agglomerative_free((Estimator*)ac);
}

TEST(test_agg_two_clusters) {
    /* Two well-separated groups */
    Matrix *X = matrix_alloc(6, 1);
    matrix_set(X, 0, 0, 1.0);
    matrix_set(X, 1, 0, 2.0);
    matrix_set(X, 2, 0, 3.0);
    matrix_set(X, 3, 0, 20.0);
    matrix_set(X, 4, 0, 21.0);
    matrix_set(X, 5, 0, 22.0);

    AgglomerativeClustering *ac = agglomerative_create(2, LINKAGE_SINGLE);
    Estimator *fitted = agglomerative_fit((Estimator*)ac, X, NULL);
    ASSERT_NOT_NULL(fitted);
    ASSERT_NOT_NULL(ac->labels);

    /* First 3 should be in same cluster, last 3 in another */
    ASSERT_EQ(ac->labels[0], ac->labels[1]);
    ASSERT_EQ(ac->labels[1], ac->labels[2]);
    ASSERT_EQ(ac->labels[3], ac->labels[4]);
    ASSERT_EQ(ac->labels[4], ac->labels[5]);
    ASSERT(ac->labels[0] != ac->labels[3]);

    matrix_free(X);
    agglomerative_free((Estimator*)ac);
}

TEST(test_agg_three_clusters) {
    Matrix *X = matrix_alloc(9, 1);
    /* Cluster A: [1, 2, 3] */
    matrix_set(X, 0, 0, 1.0);
    matrix_set(X, 1, 0, 2.0);
    matrix_set(X, 2, 0, 3.0);
    /* Cluster B: [10, 11, 12] */
    matrix_set(X, 3, 0, 10.0);
    matrix_set(X, 4, 0, 11.0);
    matrix_set(X, 5, 0, 12.0);
    /* Cluster C: [50, 51, 52] */
    matrix_set(X, 6, 0, 50.0);
    matrix_set(X, 7, 0, 51.0);
    matrix_set(X, 8, 0, 52.0);

    AgglomerativeClustering *ac = agglomerative_create(3, LINKAGE_COMPLETE);
    agglomerative_fit((Estimator*)ac, X, NULL);

    ASSERT_NOT_NULL(ac->labels);
    /* Each group of 3 should be in the same cluster */
    ASSERT_EQ(ac->labels[0], ac->labels[1]);
    ASSERT_EQ(ac->labels[1], ac->labels[2]);
    ASSERT_EQ(ac->labels[3], ac->labels[4]);
    ASSERT_EQ(ac->labels[4], ac->labels[5]);
    ASSERT_EQ(ac->labels[6], ac->labels[7]);
    ASSERT_EQ(ac->labels[7], ac->labels[8]);

    /* All three clusters should be different */
    ASSERT(ac->labels[0] != ac->labels[3]);
    ASSERT(ac->labels[3] != ac->labels[6]);
    ASSERT(ac->labels[0] != ac->labels[6]);

    matrix_free(X);
    agglomerative_free((Estimator*)ac);
}

TEST(test_agg_average_linkage) {
    Matrix *X = matrix_alloc(4, 2);
    matrix_set(X, 0, 0, 0.0);  matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 1.0);  matrix_set(X, 1, 1, 0.0);
    matrix_set(X, 2, 0, 10.0); matrix_set(X, 2, 1, 10.0);
    matrix_set(X, 3, 0, 11.0); matrix_set(X, 3, 1, 10.0);

    AgglomerativeClustering *ac = agglomerative_create(2, LINKAGE_AVERAGE);
    agglomerative_fit((Estimator*)ac, X, NULL);

    ASSERT_NOT_NULL(ac->labels);
    ASSERT_EQ(ac->labels[0], ac->labels[1]);
    ASSERT_EQ(ac->labels[2], ac->labels[3]);
    ASSERT(ac->labels[0] != ac->labels[2]);

    matrix_free(X);
    agglomerative_free((Estimator*)ac);
}

int main(void) {
    printf("=== Agglomerative Clustering Tests ===\n\n");
    RUN_TEST(test_agg_create);
    RUN_TEST(test_agg_two_clusters);
    RUN_TEST(test_agg_three_clusters);
    RUN_TEST(test_agg_average_linkage);
    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return 0;
}
