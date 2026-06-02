/**
 * test_isolation_forest.c - Tests for Isolation Forest
 */

#include "test_harness.h"
#include "isolation_forest.h"
#include "matrix.h"

TEST(test_if_create) {
    IsolationForest *ifor = isolation_forest_create(50, 0);
    ASSERT_NOT_NULL(ifor);
    ASSERT_EQ(ifor->n_trees, 50);
    isolation_forest_free((Estimator*)ifor);
}

TEST(test_if_anomaly_detection) {
    /* Normal points: centered at 5,5. Anomaly: at 50,50 */
    size_t n = 50;
    Matrix *X = matrix_alloc(n, 2);

    for (size_t i = 0; i < n - 2; i++) {
        matrix_set(X, i, 0, 5.0 + (double)(i % 10) * 0.1);
        matrix_set(X, i, 1, 5.0 + (double)(i % 7) * 0.1);
    }
    /* Two anomalies */
    matrix_set(X, n - 2, 0, 50.0);
    matrix_set(X, n - 2, 1, 50.0);
    matrix_set(X, n - 1, 0, -30.0);
    matrix_set(X, n - 1, 1, -30.0);

    IsolationForest *ifor = isolation_forest_create(100, 20);
    Estimator *fitted = isolation_forest_fit((Estimator*)ifor, X, NULL);
    ASSERT_NOT_NULL(fitted);

    /* Score samples — anomalies should have higher scores */
    Matrix *scores = isolation_forest_score_samples((Estimator*)ifor, X);
    ASSERT_NOT_NULL(scores);

    /* Anomalies (last 2) should score higher than normal points */
    double normal_score = 0.0;
    for (size_t i = 0; i < n - 2; i++) normal_score += scores->data[i];
    normal_score /= (double)(n - 2);

    double anomaly_score = (scores->data[n - 2] + scores->data[n - 1]) / 2.0;

    ASSERT(anomaly_score > normal_score);

    matrix_free(scores);
    matrix_free(X);
    isolation_forest_free((Estimator*)ifor);
}

TEST(test_if_predict_labels) {
    size_t n = 40;
    Matrix *X = matrix_alloc(n, 2);

    for (size_t i = 0; i < n - 2; i++) {
        matrix_set(X, i, 0, 3.0 + (double)(i % 8) * 0.1);
        matrix_set(X, i, 1, 3.0 + (double)(i % 5) * 0.1);
    }
    matrix_set(X, n - 2, 0, 100.0);
    matrix_set(X, n - 2, 1, 100.0);
    matrix_set(X, n - 1, 0, -80.0);
    matrix_set(X, n - 1, 1, -80.0);

    IsolationForest *ifor = isolation_forest_create(50, 20);
    isolation_forest_fit((Estimator*)ifor, X, NULL);

    Matrix *pred = isolation_forest_predict((Estimator*)ifor, X);
    ASSERT_NOT_NULL(pred);

    /* Predictions should be -1 or 1 */
    for (size_t i = 0; i < n; i++) {
        int label = (int)pred->data[i];
        ASSERT(label == -1 || label == 1);
    }

    matrix_free(pred);
    matrix_free(X);
    isolation_forest_free((Estimator*)ifor);
}

int main(void) {
    printf("=== Isolation Forest Tests ===\n\n");
    RUN_TEST(test_if_create);
    RUN_TEST(test_if_anomaly_detection);
    RUN_TEST(test_if_predict_labels);
    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return 0;
}
