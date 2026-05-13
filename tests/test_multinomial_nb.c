/**
 * @file test_multinomial_nb.c
 * @brief Unit tests for Multinomial Naive Bayes classifier
 */

#include "test_harness.h"
#include "matrix.h"
#include "naive_bayes.h"
#include "estimator.h"

TEST(test_multinomial_nb_create_free) {
    MultinomialNB *nb = multinomial_nb_create();
    ASSERT_NOT_NULL(nb);

    ASSERT_EQ(nb->base.type, MODEL_NAIVE_BAYES);
    ASSERT_EQ(nb->base.task, TASK_CLASSIFICATION);
    ASSERT_EQ(nb->base.is_fitted, 0);
    ASSERT_NEAR(nb->alpha, 1.0, 1e-12);
    ASSERT_NULL(nb->log_prior);
    ASSERT_NULL(nb->theta);
    ASSERT_EQ(nb->n_classes, 0);
    ASSERT_EQ(nb->n_features, 0);

    nb->base.free((Estimator *)nb);
}

TEST(test_multinomial_nb_text) {
    /* Simulated text classification with word count features.
     * 5 "vocabulary" words. 3 classes (topics).
     * Class 0 (sports): high counts for words 0,1
     * Class 1 (politics): high counts for words 2,3
     * Class 2 (tech): high counts for words 3,4
     */
    int N = 15;
    int F = 5;
    Matrix *X = matrix_alloc(N, F);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0 (sports): words 0,1 dominant */
    double c0[][5] = {
        {5,3,0,0,0},
        {4,4,1,0,0},
        {3,5,0,1,0},
        {6,2,0,0,1},
        {4,3,1,0,0}
    };
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < F; j++) {
            matrix_set(X, i, j, c0[i][j]);
        }
        matrix_set(y, i, 0, 0.0);
    }

    /* Class 1 (politics): words 2,3 dominant */
    double c1[][5] = {
        {0,1,5,4,0},
        {1,0,4,5,0},
        {0,0,6,3,1},
        {0,1,3,6,0},
        {1,0,4,4,1}
    };
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < F; j++) {
            matrix_set(X, 5+i, j, c1[i][j]);
        }
        matrix_set(y, 5+i, 0, 1.0);
    }

    /* Class 2 (tech): words 3,4 dominant */
    double c2[][5] = {
        {0,0,1,3,5},
        {0,1,0,4,4},
        {1,0,0,5,3},
        {0,0,1,4,5},
        {0,1,1,3,4}
    };
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < F; j++) {
            matrix_set(X, 10+i, j, c2[i][j]);
        }
        matrix_set(y, 10+i, 0, 2.0);
    }

    MultinomialNB *nb = multinomial_nb_create();
    ASSERT_NOT_NULL(nb);

    Estimator *fitted = nb->base.fit((Estimator *)nb, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(nb->base.is_fitted, 1);
    ASSERT_EQ(nb->n_classes, 3);
    ASSERT_EQ(nb->n_features, F);

    double acc = nb->base.score((const Estimator *)nb, X, y);
    ASSERT(acc > 0.7);

    nb->base.free((Estimator *)nb);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_multinomial_nb_proba) {
    /* Probabilities should be valid: in [0,1], rows sum to ~1.0 */
    int N = 9;
    int F = 4;
    Matrix *X = matrix_alloc(N, F);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* 3 classes, 3 samples each */
    double data[][4] = {
        {3,1,0,0}, {2,2,0,0}, {4,1,0,0},   /* class 0 */
        {0,0,3,1}, {0,0,2,2}, {0,0,4,1},   /* class 1 */
        {1,0,0,3}, {0,1,0,4}, {1,0,0,2}    /* class 2 */
    };
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < F; j++) {
            matrix_set(X, i, j, data[i][j]);
        }
        matrix_set(y, i, 0, (double)(i / 3));
    }

    MultinomialNB *nb = multinomial_nb_create();
    ASSERT_NOT_NULL(nb);

    nb->base.fit((Estimator *)nb, X, y);
    ASSERT_EQ(nb->base.is_fitted, 1);

    Matrix *proba = nb->base.predict_proba((const Estimator *)nb, X);
    ASSERT_NOT_NULL(proba);
    ASSERT_EQ(proba->rows, (size_t)N);
    ASSERT_EQ(proba->cols, (size_t)3);

    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int c = 0; c < 3; c++) {
            double p = matrix_get(proba, i, c);
            ASSERT(p >= 0.0 && p <= 1.0);
            row_sum += p;
        }
        ASSERT_NEAR(row_sum, 1.0, 1e-9);
    }

    matrix_free(proba);
    nb->base.free((Estimator *)nb);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_multinomial_nb_smoothing) {
    /* alpha=0 vs alpha=1 should produce different theta values */
    int N = 4;
    int F = 3;
    Matrix *X = matrix_alloc(N, F);
    Matrix *y = matrix_alloc(N, 1);
    ASSERT_NOT_NULL(X);
    ASSERT_NOT_NULL(y);

    /* Class 0: features 0,1 have counts, feature 2 is 0 */
    /* Class 1: features 1,2 have counts, feature 0 is 0 */
    double data[][3] = {
        {3,2,0}, {4,1,0},   /* class 0 */
        {0,1,3}, {0,2,4}    /* class 1 */
    };
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < F; j++) {
            matrix_set(X, i, j, data[i][j]);
        }
        matrix_set(y, i, 0, i < 2 ? 0.0 : 1.0);
    }

    /* Fit with alpha=1 (Laplace smoothing) */
    MultinomialNB *nb1 = multinomial_nb_create();
    ASSERT_NOT_NULL(nb1);
    nb1->alpha = 1.0;
    nb1->base.fit((Estimator *)nb1, X, y);

    /* Fit with alpha=0 (no smoothing) */
    MultinomialNB *nb0 = multinomial_nb_create();
    ASSERT_NOT_NULL(nb0);
    nb0->alpha = 0.0;
    nb0->base.fit((Estimator *)nb0, X, y);

    /* Theta values should differ between alpha=0 and alpha=1 */
    int differ = 0;
    for (int c = 0; c < 2; c++) {
        for (int j = 0; j < F; j++) {
            double t0 = matrix_get(nb0->theta, c, j);
            double t1 = matrix_get(nb1->theta, c, j);
            if (fabs(t0 - t1) > 1e-10) {
                differ = 1;
                break;
            }
        }
        if (differ) break;
    }
    ASSERT(differ);

    nb1->base.free((Estimator *)nb1);
    nb0->base.free((Estimator *)nb0);
    matrix_free(X);
    matrix_free(y);
}

int main(void) {
    printf("Multinomial Naive Bayes Tests\n");
    printf("=============================\n\n");

    RUN_TEST(test_multinomial_nb_create_free);
    RUN_TEST(test_multinomial_nb_text);
    RUN_TEST(test_multinomial_nb_proba);
    RUN_TEST(test_multinomial_nb_smoothing);

    TEST_SUMMARY();
}
