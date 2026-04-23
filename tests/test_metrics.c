/**
 * @file test_metrics.c
 * @brief Unit tests for evaluation metrics
 */

#include "test_harness.h"
#include "matrix.h"
#include "metrics.h"

TEST(test_mse_perfect) {
    /* Identical vectors => MSE = 0 */
    Matrix *y = matrix_alloc(3, 1);
    matrix_set(y, 0, 0, 1.0);
    matrix_set(y, 1, 0, 2.0);
    matrix_set(y, 2, 0, 3.0);

    double result = mse(y, y);
    ASSERT_NEAR(result, 0.0, 1e-10);

    matrix_free(y);
}

TEST(test_mse_known) {
    /* y_true = [1, 2, 3], y_pred = [1, 2, 4]
     * MSE = ((1-1)^2 + (2-2)^2 + (3-4)^2) / 3 = 1/3 */
    Matrix *y_true = matrix_alloc(3, 1);
    Matrix *y_pred = matrix_alloc(3, 1);

    matrix_set(y_true, 0, 0, 1.0);
    matrix_set(y_true, 1, 0, 2.0);
    matrix_set(y_true, 2, 0, 3.0);

    matrix_set(y_pred, 0, 0, 1.0);
    matrix_set(y_pred, 1, 0, 2.0);
    matrix_set(y_pred, 2, 0, 4.0);

    double result = mse(y_true, y_pred);
    ASSERT_NEAR(result, 1.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_rmse_known) {
    /* Same as above, RMSE = sqrt(1/3) */
    Matrix *y_true = matrix_alloc(3, 1);
    Matrix *y_pred = matrix_alloc(3, 1);

    matrix_set(y_true, 0, 0, 1.0);
    matrix_set(y_true, 1, 0, 2.0);
    matrix_set(y_true, 2, 0, 3.0);

    matrix_set(y_pred, 0, 0, 1.0);
    matrix_set(y_pred, 1, 0, 2.0);
    matrix_set(y_pred, 2, 0, 4.0);

    double result = rmse(y_true, y_pred);
    ASSERT_NEAR(result, sqrt(1.0 / 3.0), 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_mae_known) {
    /* y_true = [1, 2, 3], y_pred = [2, 2, 5]
     * MAE = (|1-2| + |2-2| + |3-5|) / 3 = 3/3 = 1.0 */
    Matrix *y_true = matrix_alloc(3, 1);
    Matrix *y_pred = matrix_alloc(3, 1);

    matrix_set(y_true, 0, 0, 1.0);
    matrix_set(y_true, 1, 0, 2.0);
    matrix_set(y_true, 2, 0, 3.0);

    matrix_set(y_pred, 0, 0, 2.0);
    matrix_set(y_pred, 1, 0, 2.0);
    matrix_set(y_pred, 2, 0, 5.0);

    double result = mae(y_true, y_pred);
    ASSERT_NEAR(result, 1.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_accuracy_perfect) {
    /* Perfect predictions => accuracy = 1.0 */
    Matrix *y = matrix_alloc(4, 1);
    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 1.0);
    matrix_set(y, 2, 0, 0.0);
    matrix_set(y, 3, 0, 1.0);

    double result = accuracy(y, y);
    ASSERT_NEAR(result, 1.0, 1e-10);

    matrix_free(y);
}

TEST(test_accuracy_partial) {
    /* y_true = [0, 1, 0, 1], y_pred = [0, 1, 1, 1]
     * 3/4 correct => accuracy = 0.75 */
    Matrix *y_true = matrix_alloc(4, 1);
    Matrix *y_pred = matrix_alloc(4, 1);

    matrix_set(y_true, 0, 0, 0.0);
    matrix_set(y_true, 1, 0, 1.0);
    matrix_set(y_true, 2, 0, 0.0);
    matrix_set(y_true, 3, 0, 1.0);

    matrix_set(y_pred, 0, 0, 0.0);
    matrix_set(y_pred, 1, 0, 1.0);
    matrix_set(y_pred, 2, 0, 1.0);
    matrix_set(y_pred, 3, 0, 1.0);

    double result = accuracy(y_true, y_pred);
    ASSERT_NEAR(result, 0.75, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_confusion_matrix) {
    /* y_true = [0, 0, 1, 1, 0, 1]
     * y_pred = [0, 1, 1, 0, 0, 1]
     * TP: actual=1, pred=1 => indices 2,5 => 2
     * TN: actual=0, pred=0 => indices 0,4 => 2
     * FP: actual=0, pred=1 => index 1 => 1
     * FN: actual=1, pred=0 => index 3 => 1 */
    Matrix *y_true = matrix_alloc(6, 1);
    Matrix *y_pred = matrix_alloc(6, 1);

    matrix_set(y_true, 0, 0, 0.0);
    matrix_set(y_true, 1, 0, 0.0);
    matrix_set(y_true, 2, 0, 1.0);
    matrix_set(y_true, 3, 0, 1.0);
    matrix_set(y_true, 4, 0, 0.0);
    matrix_set(y_true, 5, 0, 1.0);

    matrix_set(y_pred, 0, 0, 0.0);
    matrix_set(y_pred, 1, 0, 1.0);
    matrix_set(y_pred, 2, 0, 1.0);
    matrix_set(y_pred, 3, 0, 0.0);
    matrix_set(y_pred, 4, 0, 0.0);
    matrix_set(y_pred, 5, 0, 1.0);

    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    ASSERT_EQ(cm.tp, 2);
    ASSERT_EQ(cm.tn, 2);
    ASSERT_EQ(cm.fp, 1);
    ASSERT_EQ(cm.fn, 1);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_precision) {
    /* TP=2, FP=1 => precision = 2/3 */
    Matrix *y_true = matrix_alloc(6, 1);
    Matrix *y_pred = matrix_alloc(6, 1);

    matrix_set(y_true, 0, 0, 0.0);
    matrix_set(y_true, 1, 0, 0.0);
    matrix_set(y_true, 2, 0, 1.0);
    matrix_set(y_true, 3, 0, 1.0);
    matrix_set(y_true, 4, 0, 0.0);
    matrix_set(y_true, 5, 0, 1.0);

    matrix_set(y_pred, 0, 0, 0.0);
    matrix_set(y_pred, 1, 0, 1.0);
    matrix_set(y_pred, 2, 0, 1.0);
    matrix_set(y_pred, 3, 0, 0.0);
    matrix_set(y_pred, 4, 0, 0.0);
    matrix_set(y_pred, 5, 0, 1.0);

    double p = precision(y_true, y_pred);
    ASSERT_NEAR(p, 2.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_recall) {
    /* TP=2, FN=1 => recall = 2/3 */
    Matrix *y_true = matrix_alloc(6, 1);
    Matrix *y_pred = matrix_alloc(6, 1);

    matrix_set(y_true, 0, 0, 0.0);
    matrix_set(y_true, 1, 0, 0.0);
    matrix_set(y_true, 2, 0, 1.0);
    matrix_set(y_true, 3, 0, 1.0);
    matrix_set(y_true, 4, 0, 0.0);
    matrix_set(y_true, 5, 0, 1.0);

    matrix_set(y_pred, 0, 0, 0.0);
    matrix_set(y_pred, 1, 0, 1.0);
    matrix_set(y_pred, 2, 0, 1.0);
    matrix_set(y_pred, 3, 0, 0.0);
    matrix_set(y_pred, 4, 0, 0.0);
    matrix_set(y_pred, 5, 0, 1.0);

    double r = recall(y_true, y_pred);
    ASSERT_NEAR(r, 2.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_f1_score) {
    /* precision = 2/3, recall = 2/3
     * F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * 4/9 / (4/3) = 2/3 */
    Matrix *y_true = matrix_alloc(6, 1);
    Matrix *y_pred = matrix_alloc(6, 1);

    matrix_set(y_true, 0, 0, 0.0);
    matrix_set(y_true, 1, 0, 0.0);
    matrix_set(y_true, 2, 0, 1.0);
    matrix_set(y_true, 3, 0, 1.0);
    matrix_set(y_true, 4, 0, 0.0);
    matrix_set(y_true, 5, 0, 1.0);

    matrix_set(y_pred, 0, 0, 0.0);
    matrix_set(y_pred, 1, 0, 1.0);
    matrix_set(y_pred, 2, 0, 1.0);
    matrix_set(y_pred, 3, 0, 0.0);
    matrix_set(y_pred, 4, 0, 0.0);
    matrix_set(y_pred, 5, 0, 1.0);

    double f1 = f1_score(y_true, y_pred);
    ASSERT_NEAR(f1, 2.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_f1_perfect) {
    /* Perfect predictions => F1 = 1.0 */
    Matrix *y = matrix_alloc(4, 1);
    matrix_set(y, 0, 0, 0.0);
    matrix_set(y, 1, 0, 1.0);
    matrix_set(y, 2, 0, 0.0);
    matrix_set(y, 3, 0, 1.0);

    double f1 = f1_score(y, y);
    ASSERT_NEAR(f1, 1.0, 1e-10);

    matrix_free(y);
}

TEST(test_metrics_null_inputs) {
    /* NULL inputs should return 0.0 for metrics */
    ASSERT_NEAR(mse(NULL, NULL), 0.0, 1e-10);
    ASSERT_NEAR(mae(NULL, NULL), 0.0, 1e-10);
    ASSERT_NEAR(accuracy(NULL, NULL), 0.0, 1e-10);
}

int main(void) {
    printf("Evaluation Metrics Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_mse_perfect);
    RUN_TEST(test_mse_known);
    RUN_TEST(test_rmse_known);
    RUN_TEST(test_mae_known);
    RUN_TEST(test_accuracy_perfect);
    RUN_TEST(test_accuracy_partial);
    RUN_TEST(test_confusion_matrix);
    RUN_TEST(test_precision);
    RUN_TEST(test_recall);
    RUN_TEST(test_f1_score);
    RUN_TEST(test_f1_perfect);
    RUN_TEST(test_metrics_null_inputs);

    TEST_SUMMARY();
}
