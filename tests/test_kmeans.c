/**
 * @file test_kmeans.c
 * @brief Unit tests for k-Means clustering
 */

#include "test_harness.h"
#include "matrix.h"
#include "kmeans.h"

TEST(test_kmeans_fit_basic) {
    /* Two well-separated clusters in 1D:
     * Cluster A: [1, 2, 3]  Cluster B: [20, 21, 22]
     */
    Matrix *X = matrix_alloc(6, 1);
    matrix_set(X, 0, 0, 1.0);
    matrix_set(X, 1, 0, 2.0);
    matrix_set(X, 2, 0, 3.0);
    matrix_set(X, 3, 0, 20.0);
    matrix_set(X, 4, 0, 21.0);
    matrix_set(X, 5, 0, 22.0);

    KMeansModel *model = kmeans_fit(X, 2, 100, 42);
    ASSERT_NOT_NULL(model);
    ASSERT_EQ(model->k, 2);
    ASSERT_NOT_NULL(model->centroids);
    ASSERT_EQ(model->centroids->rows, 2);
    ASSERT_EQ(model->centroids->cols, 1);

    /* Centroids should be near 2 and 21 */
    double c0 = matrix_get(model->centroids, 0, 0);
    double c1 = matrix_get(model->centroids, 1, 0);
    /* One centroid near 2, the other near 21 (order may vary) */
    int ok = 0;
    if ((fabs(c0 - 2.0) < 0.5 && fabs(c1 - 21.0) < 0.5) ||
        (fabs(c0 - 21.0) < 0.5 && fabs(c1 - 2.0) < 0.5)) {
        ok = 1;
    }
    ASSERT(ok);

    kmeans_free(model);
    matrix_free(X);
}

TEST(test_kmeans_predict_labels) {
    /* Same separated clusters, test that predict assigns correct labels */
    Matrix *X = matrix_alloc(6, 1);
    matrix_set(X, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0);
    matrix_set(X, 2, 0, 2.0);
    matrix_set(X, 3, 0, 100.0);
    matrix_set(X, 4, 0, 101.0);
    matrix_set(X, 5, 0, 102.0);

    KMeansModel *model = kmeans_fit(X, 2, 100, 42);
    ASSERT_NOT_NULL(model);

    Matrix *labels = kmeans_predict(model, X);
    ASSERT_NOT_NULL(labels);
    ASSERT_EQ(labels->rows, 6);

    /* First 3 points should share one label, last 3 share another */
    int label0 = (int)matrix_get(labels, 0, 0);
    int label1 = (int)matrix_get(labels, 1, 0);
    int label2 = (int)matrix_get(labels, 2, 0);
    int label3 = (int)matrix_get(labels, 3, 0);
    int label4 = (int)matrix_get(labels, 4, 0);
    int label5 = (int)matrix_get(labels, 5, 0);

    /* All first cluster same label */
    ASSERT_EQ(label0, label1);
    ASSERT_EQ(label1, label2);
    /* All second cluster same label */
    ASSERT_EQ(label3, label4);
    ASSERT_EQ(label4, label5);
    /* Different clusters have different labels */
    ASSERT(label0 != label3);

    matrix_free(labels);
    kmeans_free(model);
    matrix_free(X);
}

TEST(test_kmeans_predict_new_points) {
    /* Cluster and then predict on new unseen data */
    Matrix *X_train = matrix_alloc(4, 1);
    matrix_set(X_train, 0, 0, 0.0);
    matrix_set(X_train, 1, 0, 1.0);
    matrix_set(X_train, 2, 0, 50.0);
    matrix_set(X_train, 3, 0, 51.0);

    KMeansModel *model = kmeans_fit(X_train, 2, 100, 42);
    ASSERT_NOT_NULL(model);

    /* Predict on new points */
    Matrix *X_test = matrix_alloc(2, 1);
    matrix_set(X_test, 0, 0, 0.5);   /* near cluster 0 */
    matrix_set(X_test, 1, 0, 50.5);  /* near cluster 1 */

    Matrix *pred = kmeans_predict(model, X_test);
    ASSERT_NOT_NULL(pred);

    /* They should be assigned to different clusters */
    ASSERT((int)matrix_get(pred, 0, 0) != (int)matrix_get(pred, 1, 0));

    matrix_free(pred);
    matrix_free(X_test);
    kmeans_free(model);
    matrix_free(X_train);
}

TEST(test_kmeans_2d_clusters) {
    /* Two 2D clusters:
     * Cluster A around (0,0), Cluster B around (10,10)
     */
    Matrix *X = matrix_alloc(6, 2);
    /* Cluster A */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0);
    /* Cluster B */
    matrix_set(X, 3, 0, 10.0); matrix_set(X, 3, 1, 10.0);
    matrix_set(X, 4, 0, 11.0); matrix_set(X, 4, 1, 10.0);
    matrix_set(X, 5, 0, 10.0); matrix_set(X, 5, 1, 11.0);

    KMeansModel *model = kmeans_fit(X, 2, 100, 42);
    ASSERT_NOT_NULL(model);
    ASSERT_EQ(model->centroids->cols, 2);

    /* Centroids should be near (0.33, 0.33) and (10.33, 10.33) */
    double c0x = matrix_get(model->centroids, 0, 0);
    double c0y = matrix_get(model->centroids, 0, 1);
    double c1x = matrix_get(model->centroids, 1, 0);
    double c1y = matrix_get(model->centroids, 1, 1);

    int ok = 0;
    if ((fabs(c0x - 0.333) < 1.0 && fabs(c1x - 10.333) < 1.0) ||
        (fabs(c0x - 10.333) < 1.0 && fabs(c1x - 0.333) < 1.0)) {
        ok = 1;
    }
    ASSERT(ok);

    kmeans_free(model);
    matrix_free(X);
}

TEST(test_kmeans_null_inputs) {
    /* NULL inputs should return NULL */
    KMeansModel *model = kmeans_fit(NULL, 2, 100, 42);
    ASSERT_NULL(model);

    Matrix *pred = kmeans_predict(NULL, NULL);
    ASSERT_NULL(pred);
}

int main(void) {
    printf("k-Means Clustering Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_kmeans_fit_basic);
    RUN_TEST(test_kmeans_predict_labels);
    RUN_TEST(test_kmeans_predict_new_points);
    RUN_TEST(test_kmeans_2d_clusters);
    RUN_TEST(test_kmeans_null_inputs);

    TEST_SUMMARY();
}
