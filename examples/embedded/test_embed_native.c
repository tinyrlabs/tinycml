/**
 * @file test_embed_native.c
 * @brief Native test for tinycml-embed: validates pool allocator +
 *        feature flag guards + cross-compilation compatibility
 *
 * Compile & run on host:
 *   make -f embed/Makefile.embed test_native
 *
 * Tests:
 *   1. Pool init/stats/reset
 *   2. KNN training + prediction via pool
 *   3. GaussianNB training + prediction via pool
 *   4. Pool exhaustion detection
 *   5. Multiple reset cycles
 *   6. Feature flag isolation
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Configuration: KNN + Naive Bayes + Linear Regression
 * ============================================================ */
#define CML_ENABLE_KNN
#define CML_ENABLE_NAIVE_BAYES
#define CML_ENABLE_LINEAR_REGRESSION
#define CML_USE_POOL
/* Pool size comes from Makefile (-DCML_POOL_SIZE=N) */

#include "cml_config.h"
#include "cml_pool.h"
#include "tinycml.h"

/* ============================================================
 * Test helpers
 * ============================================================ */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        printf("  FAIL: %s (line %d)\n", name, __LINE__); \
        tests_failed++; \
    } else { \
        printf("  PASS: %s\n", name); \
        tests_passed++; \
    } \
} while(0)

#define TEST_NEAR(name, a, b, eps) TEST(name, fabs((a) - (b)) < (eps))

/* ============================================================
 * Test 1: Pool basics
 * ============================================================ */
static void test_pool_basics(void) {
    printf("\n--- Test 1: Pool Basics ---\n");

    CMLPoolStats s = cml_pool_stats();
    TEST("pool starts uninitialized", s.init_called == 0);

    int ret = cml_pool_init();
    TEST("pool init returns 0", ret == 0);

    s = cml_pool_stats();
    TEST("pool total > 0", s.total_size > 0);
    TEST("pool used == 0 after init", s.used == 0);

    void *p1 = cml_pool_alloc(32);
    TEST("basic alloc returns non-NULL", p1 != NULL);
    TEST("pointer is 8-byte aligned", ((size_t)p1 & 7) == 0);

    s = cml_pool_stats();
    TEST("used >= 32 after alloc", s.used >= 32);

    cml_pool_reset();
    s = cml_pool_stats();
    TEST("used == 0 after reset", s.used == 0);
}

/* ============================================================
 * Test 2: Pool calloc/realloc
 * ============================================================ */
static void test_pool_calloc_realloc(void) {
    printf("\n--- Test 2: Pool calloc/realloc ---\n");

    cml_pool_init();

    int *arr = (int*)cml_pool_calloc(10, sizeof(int));
    TEST("calloc returns non-NULL", arr != NULL);
    TEST("calloc zero-initialized", arr[0] == 0 && arr[9] == 0);

    arr[3] = 42;

    int *arr2 = (int*)cml_pool_realloc(arr, 20 * sizeof(int));
    TEST("realloc returns non-NULL", arr2 != NULL);
    TEST("realloc preserves data", arr2[3] == 42);

    cml_pool_reset();
}

/* ============================================================
 * Test 3: KNN via pool
 * ============================================================ */
static void test_knn_pool(void) {
    printf("\n--- Test 3: KNN Training & Prediction via Pool ---\n");

    cml_pool_init();

    /* Training data: simple AND gate pattern (2 features) */
    double X_data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double y_data[4]    = {0, 1, 1, 1};

    Matrix *X = matrix_alloc(4, 2);
    Matrix *y = matrix_alloc(4, 1);
    TEST("KNN: X alloc", X != NULL);
    TEST("KNN: y alloc", y != NULL);

    for (int i = 0; i < 4; i++) {
        matrix_set(X, i, 0, X_data[i][0]);
        matrix_set(X, i, 1, X_data[i][1]);
        matrix_set(y, i, 0, y_data[i]);
    }

    KNNModel *knn = knn_fit(X, y, 1);  /* k=1 for deterministic prediction */
    TEST("KNN: fit succeeded", knn != NULL);

    /* Predict */
    Matrix *X_test = matrix_alloc(1, 2);
    matrix_set(X_test, 0, 0, 1.0);
    matrix_set(X_test, 0, 1, 1.0);
    Matrix *pred = knn_predict(knn, X_test);
    TEST("KNN: predict returns non-NULL", pred != NULL);
    TEST("KNN: 1 AND 1 = 1", pred->data[0] == 1.0);

    matrix_set(X_test, 0, 0, 0.0);
    matrix_set(X_test, 0, 1, 0.0);
    Matrix *pred2 = knn_predict(knn, X_test);
    TEST("KNN: 0 AND 0 = 0", pred2->data[0] == 0.0);

    matrix_free(X_test);
    matrix_free(pred);
    matrix_free(pred2);

    /* Cleanup */
    knn_free(knn);
    matrix_free(X);
    matrix_free(y);

    cml_pool_reset();
}

/* ============================================================
 * Test 4: GaussianNB via pool
 * ============================================================ */
static void test_nb_pool(void) {
    printf("\n--- Test 4: GaussianNB Training & Prediction via Pool ---\n");

    cml_pool_init();

    /* Simple binary classification on 1D data */
    double X_data[6][1] = {{1.0}, {1.5}, {2.0}, {8.0}, {9.0}, {10.0}};
    double y_data[6]    = {0, 0, 0, 1, 1, 1};

    Matrix *X = matrix_alloc(6, 1);
    Matrix *y = matrix_alloc(6, 1);
    TEST("NB: X alloc", X != NULL);
    TEST("NB: y alloc", y != NULL);

    for (int i = 0; i < 6; i++) {
        matrix_set(X, i, 0, X_data[i][0]);
        matrix_set(y, i, 0, y_data[i]);
    }

    GaussianNaiveBayes *nb = gaussian_nb_create();
    TEST("NB: create succeeded", nb != NULL);

    Estimator *fitted = nb->base.fit((Estimator*)nb, X, y);
    TEST("NB: fit succeeded", fitted != NULL);

    /* Predict a known sample */
    Matrix *X_test = matrix_alloc(1, 1);
    matrix_set(X_test, 0, 0, 1.2);
    Matrix *pred = nb->base.predict((const Estimator*)nb, X_test);
    TEST("NB: predict returns non-NULL", pred != NULL);
    TEST("NB: 1.2 -> class 0", pred->data[0] == 0.0);

    matrix_set(X_test, 0, 0, 9.5);
    Matrix *pred2 = nb->base.predict((const Estimator*)nb, X_test);
    TEST("NB: 9.5 -> class 1", pred2->data[0] == 1.0);

    /* Score */
    double score = nb->base.score((const Estimator*)nb, X, y);
    TEST("NB: perfect training score", score == 1.0);

    matrix_free(X_test);
    matrix_free(pred);
    matrix_free(pred2);
    nb->base.free((Estimator*)nb);
    matrix_free(X);
    matrix_free(y);

    cml_pool_reset();
}

/* ============================================================
 * Test 5: Pool exhaustion
 * ============================================================ */
static void test_pool_exhaustion(void) {
    printf("\n--- Test 5: Pool Exhaustion Detection ---\n");

    cml_pool_init();

    CMLPoolStats s = cml_pool_stats();
    TEST("exhaust: pool is initialized", s.init_called == 1);

    /* Allocate almost all pool (use dynamic size from stats) */
    size_t pool_size = s.total_size;
    void *big = cml_pool_alloc(pool_size - 64);
    TEST("exhaust: large alloc succeeds", big != NULL);

    /* This should fail */
    void *fail = cml_pool_alloc(128);
    TEST("exhaust: overflow returns NULL", fail == NULL);

    cml_pool_reset();

    /* After reset, allocation should work again */
    void *after = cml_pool_alloc(128);
    TEST("exhaust: after reset alloc works", after != NULL);
}

/* ============================================================
 * Test 6: Multiple reset cycles
 * ============================================================ */
static void test_reset_cycles(void) {
    printf("\n--- Test 6: Multiple Reset Cycles ---\n");

    for (int cycle = 0; cycle < 10; cycle++) {
        cml_pool_init();

        void *p = cml_pool_alloc(128);
        TEST("cycle: pointer valid", p != NULL);

        CMLPoolStats s = cml_pool_stats();
        TEST("cycle: peak tracking", s.peak_used >= s.used);

        cml_pool_reset();
    }

    CMLPoolStats s = cml_pool_stats();
    TEST("cycle: after all resets, used == 0", s.used == 0);
}

/* ============================================================
 * Test 7: Linear Regression via pool
 * ============================================================ */
static void test_linear_regression_pool(void) {
    printf("\n--- Test 7: Linear Regression via Pool ---\n");

    cml_pool_init();

    /* y = 2x + 1 */
    double X_data[5] = {0, 1, 2, 3, 4};
    double y_data[5] = {1, 3, 5, 7, 9};

    Matrix *X = matrix_alloc(5, 1);
    Matrix *y = matrix_alloc(5, 1);
    for (int i = 0; i < 5; i++) {
        matrix_set(X, i, 0, X_data[i]);
        matrix_set(y, i, 0, y_data[i]);
    }

    LinearRegression *model = linear_regression_create(LINREG_SOLVER_AUTO);
    TEST("LR: create", model != NULL);
    Estimator *fitted = model->base.fit((Estimator*)model, X, y);
    TEST("LR: fit", fitted != NULL);

    Matrix *pred = model->base.predict((const Estimator*)model, X);
    TEST("LR: predict", pred != NULL);

    double score = model->base.score((const Estimator*)model, X, y);
    TEST("LR: R2 close to 1.0", fabs(score - 1.0) < 0.01);

    matrix_free(pred);
    model->base.free((Estimator*)model);
    matrix_free(X);
    matrix_free(y);
    cml_pool_reset();
}

/* ============================================================
 * main
 * ============================================================ */
int main(void) {
    printf("========================================\n");
    printf("  tinycml-embed Native Test Suite\n");
    printf("  Pool: %d bytes, Algos: KNN+NB+LR\n", CML_POOL_SIZE);
    printf("========================================\n");

    test_pool_basics();
    test_pool_calloc_realloc();
    test_pool_exhaustion();
    test_reset_cycles();
    test_knn_pool();
    test_nb_pool();
    test_linear_regression_pool();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
