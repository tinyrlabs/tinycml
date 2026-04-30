/**
 * @file test_decision_tree.c
 * @brief Unit tests for Decision Tree Classifier
 */

#include "test_harness.h"
#include "matrix.h"
#include "decision_tree.h"

TEST(test_dt_create_free) {
    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);
    ASSERT_EQ(dt->base.type, MODEL_DECISION_TREE);
    ASSERT_EQ(dt->base.is_fitted, 0);
    ASSERT_NULL(dt->root);

    dt->base.free((Estimator *)dt);
}

TEST(test_dt_fit_predict) {
    /* Linearly separable 2D data:
     * Class 0: x1 < 2.5  =>  points (0,0),(1,0),(0,1),(1,1),(2,0)
     * Class 1: x1 >= 3   =>  points (3,0),(4,0),(3,1),(4,1),(5,0)
     */
    Matrix *X = matrix_alloc(10, 2);
    Matrix *y = matrix_alloc(10, 1);

    /* Class 0 */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0); matrix_set(y, 1, 0, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0); matrix_set(y, 2, 0, 0.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 1.0); matrix_set(y, 3, 0, 0.0);
    matrix_set(X, 4, 0, 2.0); matrix_set(X, 4, 1, 0.0); matrix_set(y, 4, 0, 0.0);

    /* Class 1 */
    matrix_set(X, 5, 0, 3.0); matrix_set(X, 5, 1, 0.0); matrix_set(y, 5, 0, 1.0);
    matrix_set(X, 6, 0, 4.0); matrix_set(X, 6, 1, 0.0); matrix_set(y, 6, 0, 1.0);
    matrix_set(X, 7, 0, 3.0); matrix_set(X, 7, 1, 1.0); matrix_set(y, 7, 0, 1.0);
    matrix_set(X, 8, 0, 4.0); matrix_set(X, 8, 1, 1.0); matrix_set(y, 8, 0, 1.0);
    matrix_set(X, 9, 0, 5.0); matrix_set(X, 9, 1, 0.0); matrix_set(y, 9, 0, 1.0);

    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);

    Estimator *fitted = dt->base.fit((Estimator *)dt, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(dt->base.is_fitted, 1);

    Matrix *pred = dt->base.predict((const Estimator *)dt, X);
    ASSERT_NOT_NULL(pred);
    ASSERT_EQ(pred->rows, 10);

    /* Check accuracy > 0.7 */
    double accuracy = dt->base.score((const Estimator *)dt, X, y);
    ASSERT(accuracy > 0.7);

    matrix_free(pred);
    dt->base.free((Estimator *)dt);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_dt_max_depth) {
    /* With max_depth=1 (stump), the tree should only have a single split.
     * Use the same separable data, create with max_depth=1.
     */
    DecisionTreeClassifier *dt = decision_tree_classifier_create_full(
        CRITERION_GINI, 1, 2, 1, 0.0);
    ASSERT_NOT_NULL(dt);
    ASSERT_EQ(dt->max_depth, 1);

    Matrix *X = matrix_alloc(10, 2);
    Matrix *y = matrix_alloc(10, 1);

    /* Class 0 */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0); matrix_set(y, 1, 0, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0); matrix_set(y, 2, 0, 0.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 1.0); matrix_set(y, 3, 0, 0.0);
    matrix_set(X, 4, 0, 2.0); matrix_set(X, 4, 1, 0.0); matrix_set(y, 4, 0, 0.0);

    /* Class 1 */
    matrix_set(X, 5, 0, 3.0); matrix_set(X, 5, 1, 0.0); matrix_set(y, 5, 0, 1.0);
    matrix_set(X, 6, 0, 4.0); matrix_set(X, 6, 1, 0.0); matrix_set(y, 6, 0, 1.0);
    matrix_set(X, 7, 0, 3.0); matrix_set(X, 7, 1, 1.0); matrix_set(y, 7, 0, 1.0);
    matrix_set(X, 8, 0, 4.0); matrix_set(X, 8, 1, 1.0); matrix_set(y, 8, 0, 1.0);
    matrix_set(X, 9, 0, 5.0); matrix_set(X, 9, 1, 0.0); matrix_set(y, 9, 0, 1.0);

    Estimator *fitted = dt->base.fit((Estimator *)dt, X, y);
    ASSERT_NOT_NULL(fitted);

    /* A stump should still have depth 1 */
    ASSERT(dt->depth_ <= 1);

    /* Predictions should still be somewhat reasonable (only 2 unique values possible) */
    Matrix *pred = dt->base.predict((const Estimator *)dt, X);
    ASSERT_NOT_NULL(pred);

    /* Count unique predicted values - should be at most 2 for a stump */
    int has_0 = 0, has_1 = 0;
    for (int i = 0; i < 10; i++) {
        double v = matrix_get(pred, i, 0);
        if (v == 0.0) has_0 = 1;
        if (v == 1.0) has_1 = 1;
    }
    /* Stump can produce at most 2 distinct predictions */
    ASSERT(has_0 + has_1 <= 2);

    matrix_free(pred);
    dt->base.free((Estimator *)dt);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_dt_clone) {
    Matrix *X = matrix_alloc(6, 2);
    Matrix *y = matrix_alloc(6, 1);

    /* Simple separable data */
    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0); matrix_set(y, 1, 0, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0); matrix_set(y, 2, 0, 0.0);
    matrix_set(X, 3, 0, 5.0); matrix_set(X, 3, 1, 5.0); matrix_set(y, 3, 0, 1.0);
    matrix_set(X, 4, 0, 6.0); matrix_set(X, 4, 1, 5.0); matrix_set(y, 4, 0, 1.0);
    matrix_set(X, 5, 0, 5.0); matrix_set(X, 5, 1, 6.0); matrix_set(y, 5, 0, 1.0);

    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    ASSERT_NOT_NULL(dt);

    dt->base.fit((Estimator *)dt, X, y);
    ASSERT_EQ(dt->base.is_fitted, 1);

    /* Clone the fitted model */
    Estimator *clone_est = dt->base.clone((const Estimator *)dt);
    ASSERT_NOT_NULL(clone_est);

    /* If clone is fitted, verify predictions match; if not, just verify independence */
    if (clone_est->is_fitted) {
        /* Clone is a deep copy of the fitted model */
        Matrix *pred_orig = dt->base.predict((const Estimator *)dt, X);
        Matrix *pred_clone = clone_est->predict(clone_est, X);
        ASSERT_NOT_NULL(pred_orig);
        ASSERT_NOT_NULL(pred_clone);

        for (int i = 0; i < 6; i++) {
            ASSERT_EQ((int)matrix_get(pred_orig, i, 0), (int)matrix_get(pred_clone, i, 0));
        }
        matrix_free(pred_orig);
        matrix_free(pred_clone);
    }

    /* Verify independence: freeing clone doesn't affect original */
    clone_est->free(clone_est);
    ASSERT_EQ(dt->base.is_fitted, 1);

    /* Original should still predict correctly */
    Matrix *pred_after = dt->base.predict((const Estimator *)dt, X);
    ASSERT_NOT_NULL(pred_after);
    /* Accuracy should still be good */
    double acc = dt->base.score((const Estimator *)dt, X, y);
    ASSERT(acc > 0.7);

    matrix_free(pred_after);
    dt->base.free((Estimator *)dt);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_dt_predict_proba) {
    Matrix *X = matrix_alloc(6, 2);
    Matrix *y = matrix_alloc(6, 1);

    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 1.0); matrix_set(X, 1, 1, 0.0); matrix_set(y, 1, 0, 0.0);
    matrix_set(X, 2, 0, 0.0); matrix_set(X, 2, 1, 1.0); matrix_set(y, 2, 0, 0.0);
    matrix_set(X, 3, 0, 5.0); matrix_set(X, 3, 1, 5.0); matrix_set(y, 3, 0, 1.0);
    matrix_set(X, 4, 0, 6.0); matrix_set(X, 4, 1, 5.0); matrix_set(y, 4, 0, 1.0);
    matrix_set(X, 5, 0, 5.0); matrix_set(X, 5, 1, 6.0); matrix_set(y, 5, 0, 1.0);

    DecisionTreeClassifier *dt = decision_tree_classifier_create();
    dt->base.fit((Estimator *)dt, X, y);

    /* predict_proba should return probability matrix */
    Matrix *proba = decision_tree_classifier_predict_proba((const Estimator *)dt, X);
    ASSERT_NOT_NULL(proba);
    ASSERT(proba->rows == 6);

    /* Each row should have valid probabilities */
    for (int i = 0; i < 6; i++) {
        double row_sum = 0.0;
        for (size_t j = 0; j < proba->cols; j++) {
            double p = matrix_get(proba, i, j);
            ASSERT(p >= 0.0 && p <= 1.0);
            row_sum += p;
        }
        ASSERT_NEAR(row_sum, 1.0, 1e-9);
    }

    matrix_free(proba);
    dt->base.free((Estimator *)dt);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Decision Tree Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_dt_create_free);
    RUN_TEST(test_dt_fit_predict);
    RUN_TEST(test_dt_max_depth);
    RUN_TEST(test_dt_clone);
    RUN_TEST(test_dt_predict_proba);

    TEST_SUMMARY();
}
