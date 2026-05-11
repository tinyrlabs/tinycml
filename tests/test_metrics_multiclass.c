/**
 * @file test_metrics_multiclass.c
 * @brief Unit tests for multi-class metrics
 */

#include "test_harness.h"
#include "matrix.h"
#include "metrics.h"

TEST(test_confusion_matrix_multi_3class) {
    /* 3 classes (0, 1, 2), 9 samples
     * y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
     * y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0]
     *
     * Confusion matrix (rows=true, cols=pred):
     *        pred0  pred1  pred2
     * true0:  2      1      0
     * true1:  0      2      1
     * true2:  1      0      2
     */
    Matrix *y_true = matrix_alloc(9, 1);
    Matrix *y_pred = matrix_alloc(9, 1);
    ASSERT_NOT_NULL(y_true);
    ASSERT_NOT_NULL(y_pred);

    double t[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    double p[] = {0, 0, 1, 1, 1, 2, 2, 2, 0};
    for (int i = 0; i < 9; i++) {
        y_true->data[i] = t[i];
        y_pred->data[i] = p[i];
    }

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    ASSERT_NOT_NULL(cm);
    ASSERT_EQ(cm->rows, 3);
    ASSERT_EQ(cm->cols, 3);

    /* Row 0 (true=0): [2, 1, 0] */
    ASSERT_NEAR(matrix_get(cm, 0, 0), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 0, 1), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 0, 2), 0.0, 1e-10);

    /* Row 1 (true=1): [0, 2, 1] */
    ASSERT_NEAR(matrix_get(cm, 1, 0), 0.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 1, 1), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 1, 2), 1.0, 1e-10);

    /* Row 2 (true=2): [1, 0, 2] */
    ASSERT_NEAR(matrix_get(cm, 2, 0), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 2, 1), 0.0, 1e-10);
    ASSERT_NEAR(matrix_get(cm, 2, 2), 2.0, 1e-10);

    matrix_free(cm);
    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_precision_macro) {
    /* Same data as above:
     * Class 0: TP=2, FP=1 (col sum=3) => prec = 2/3
     * Class 1: TP=2, FP=1 (col sum=3) => prec = 2/3
     * Class 2: TP=2, FP=1 (col sum=3) => prec = 2/3
     * Macro precision = (2/3 + 2/3 + 2/3) / 3 = 2/3
     */
    Matrix *y_true = matrix_alloc(9, 1);
    Matrix *y_pred = matrix_alloc(9, 1);
    ASSERT_NOT_NULL(y_true);
    ASSERT_NOT_NULL(y_pred);

    double t[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    double p[] = {0, 0, 1, 1, 1, 2, 2, 2, 0};
    for (int i = 0; i < 9; i++) {
        y_true->data[i] = t[i];
        y_pred->data[i] = p[i];
    }

    double pm = precision_macro(y_true, y_pred, 3);
    ASSERT_NEAR(pm, 2.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_f1_macro) {
    /* Same data:
     * Class 0: prec=2/3, recall=2/3 => F1=2/3
     * Class 1: prec=2/3, recall=2/3 => F1=2/3
     * Class 2: prec=2/3, recall=2/3 => F1=2/3
     * Macro F1 = 2/3
     */
    Matrix *y_true = matrix_alloc(9, 1);
    Matrix *y_pred = matrix_alloc(9, 1);
    ASSERT_NOT_NULL(y_true);
    ASSERT_NOT_NULL(y_pred);

    double t[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    double p[] = {0, 0, 1, 1, 1, 2, 2, 2, 0};
    for (int i = 0; i < 9; i++) {
        y_true->data[i] = t[i];
        y_pred->data[i] = p[i];
    }

    double f1m = f1_score_macro(y_true, y_pred, 3);
    ASSERT_NEAR(f1m, 2.0 / 3.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_f1_weighted) {
    /* Imbalanced scenario:
     * y_true = [0, 0, 0, 0, 1, 1, 2]  (support: 4, 2, 1)
     * y_pred = [0, 0, 0, 1, 1, 0, 2]
     *
     * CM:
     *        pred0  pred1  pred2
     * true0:  3      1      0
     * true1:  1      1      0
     * true2:  0      0      1
     *
     * Class 0: prec=3/4, recall=3/4, F1=3/4
     * Class 1: prec=1/2, recall=1/2, F1=1/2
     * Class 2: prec=1/1, recall=1/1, F1=1
     *
     * Weighted = (3/4*4 + 1/2*2 + 1*1) / 7 = (3 + 1 + 1) / 7 = 5/7
     */
    Matrix *y_true = matrix_alloc(7, 1);
    Matrix *y_pred = matrix_alloc(7, 1);
    ASSERT_NOT_NULL(y_true);
    ASSERT_NOT_NULL(y_pred);

    double t[] = {0, 0, 0, 0, 1, 1, 2};
    double p[] = {0, 0, 0, 1, 1, 0, 2};
    for (int i = 0; i < 7; i++) {
        y_true->data[i] = t[i];
        y_pred->data[i] = p[i];
    }

    double f1w = f1_score_weighted(y_true, y_pred, 3);
    ASSERT_NEAR(f1w, 5.0 / 7.0, 1e-10);

    matrix_free(y_true);
    matrix_free(y_pred);
}

TEST(test_confusion_matrix_perfect) {
    /* Perfect predictions => diagonal matrix */
    Matrix *y_true = matrix_alloc(9, 1);
    Matrix *y_pred = matrix_alloc(9, 1);
    ASSERT_NOT_NULL(y_true);
    ASSERT_NOT_NULL(y_pred);

    double t[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    for (int i = 0; i < 9; i++) {
        y_true->data[i] = t[i];
        y_pred->data[i] = t[i];  /* perfect prediction */
    }

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    ASSERT_NOT_NULL(cm);

    /* Diagonal should be 3, off-diagonal should be 0 */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                ASSERT_NEAR(matrix_get(cm, i, j), 3.0, 1e-10);
            } else {
                ASSERT_NEAR(matrix_get(cm, i, j), 0.0, 1e-10);
            }
        }
    }

    /* Macro F1 should be 1.0 for perfect predictions */
    double f1m = f1_score_macro(y_true, y_pred, 3);
    ASSERT_NEAR(f1m, 1.0, 1e-10);

    double f1w = f1_score_weighted(y_true, y_pred, 3);
    ASSERT_NEAR(f1w, 1.0, 1e-10);

    matrix_free(cm);
    matrix_free(y_true);
    matrix_free(y_pred);
}

int main(void) {
    printf("Multi-class Metrics Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_confusion_matrix_multi_3class);
    RUN_TEST(test_precision_macro);
    RUN_TEST(test_f1_macro);
    RUN_TEST(test_f1_weighted);
    RUN_TEST(test_confusion_matrix_perfect);

    TEST_SUMMARY();
}
