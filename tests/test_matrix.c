/**
 * @file test_matrix.c
 * @brief Unit tests for matrix operations
 */

#include "test_harness.h"
#include "matrix.h"

TEST(test_matrix_alloc_free) {
    Matrix *m = matrix_alloc(3, 4);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->rows, 3);
    ASSERT_EQ(m->cols, 4);

    /* Check zero initialization */
    for (size_t i = 0; i < 12; i++) {
        ASSERT_NEAR(m->data[i], 0.0, 1e-10);
    }

    matrix_free(m);
}

TEST(test_matrix_get_set) {
    Matrix *m = matrix_alloc(2, 3);
    ASSERT_NOT_NULL(m);

    matrix_set(m, 0, 0, 1.0);
    matrix_set(m, 0, 1, 2.0);
    matrix_set(m, 0, 2, 3.0);
    matrix_set(m, 1, 0, 4.0);
    matrix_set(m, 1, 1, 5.0);
    matrix_set(m, 1, 2, 6.0);

    ASSERT_NEAR(matrix_get(m, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(m, 0, 1), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(m, 0, 2), 3.0, 1e-10);
    ASSERT_NEAR(matrix_get(m, 1, 0), 4.0, 1e-10);
    ASSERT_NEAR(matrix_get(m, 1, 1), 5.0, 1e-10);
    ASSERT_NEAR(matrix_get(m, 1, 2), 6.0, 1e-10);

    matrix_free(m);
}

TEST(test_matrix_copy) {
    Matrix *m = matrix_alloc(2, 2);
    matrix_set(m, 0, 0, 1.0);
    matrix_set(m, 0, 1, 2.0);
    matrix_set(m, 1, 0, 3.0);
    matrix_set(m, 1, 1, 4.0);

    Matrix *copy = matrix_copy(m);
    ASSERT_NOT_NULL(copy);
    ASSERT_EQ(copy->rows, 2);
    ASSERT_EQ(copy->cols, 2);

    ASSERT_NEAR(matrix_get(copy, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(copy, 0, 1), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(copy, 1, 0), 3.0, 1e-10);
    ASSERT_NEAR(matrix_get(copy, 1, 1), 4.0, 1e-10);

    /* Verify deep copy */
    matrix_set(m, 0, 0, 99.0);
    ASSERT_NEAR(matrix_get(copy, 0, 0), 1.0, 1e-10);

    matrix_free(m);
    matrix_free(copy);
}

TEST(test_matrix_add) {
    Matrix *a = matrix_alloc(2, 2);
    Matrix *b = matrix_alloc(2, 2);

    matrix_set(a, 0, 0, 1.0);
    matrix_set(a, 0, 1, 2.0);
    matrix_set(a, 1, 0, 3.0);
    matrix_set(a, 1, 1, 4.0);

    matrix_set(b, 0, 0, 5.0);
    matrix_set(b, 0, 1, 6.0);
    matrix_set(b, 1, 0, 7.0);
    matrix_set(b, 1, 1, 8.0);

    Matrix *c = matrix_add(a, b);
    ASSERT_NOT_NULL(c);

    ASSERT_NEAR(matrix_get(c, 0, 0), 6.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 0, 1), 8.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 1, 0), 10.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 1, 1), 12.0, 1e-10);

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

TEST(test_matrix_matmul) {
    /* A = [1 2; 3 4], B = [5 6; 7 8] */
    /* A * B = [19 22; 43 50] */
    Matrix *a = matrix_alloc(2, 2);
    Matrix *b = matrix_alloc(2, 2);

    matrix_set(a, 0, 0, 1.0);
    matrix_set(a, 0, 1, 2.0);
    matrix_set(a, 1, 0, 3.0);
    matrix_set(a, 1, 1, 4.0);

    matrix_set(b, 0, 0, 5.0);
    matrix_set(b, 0, 1, 6.0);
    matrix_set(b, 1, 0, 7.0);
    matrix_set(b, 1, 1, 8.0);

    Matrix *c = matrix_matmul(a, b);
    ASSERT_NOT_NULL(c);

    ASSERT_NEAR(matrix_get(c, 0, 0), 19.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 0, 1), 22.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 1, 0), 43.0, 1e-10);
    ASSERT_NEAR(matrix_get(c, 1, 1), 50.0, 1e-10);

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

TEST(test_matrix_transpose) {
    Matrix *m = matrix_alloc(2, 3);
    matrix_set(m, 0, 0, 1.0);
    matrix_set(m, 0, 1, 2.0);
    matrix_set(m, 0, 2, 3.0);
    matrix_set(m, 1, 0, 4.0);
    matrix_set(m, 1, 1, 5.0);
    matrix_set(m, 1, 2, 6.0);

    Matrix *t = matrix_transpose(m);
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->rows, 3);
    ASSERT_EQ(t->cols, 2);

    ASSERT_NEAR(matrix_get(t, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(t, 0, 1), 4.0, 1e-10);
    ASSERT_NEAR(matrix_get(t, 1, 0), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(t, 1, 1), 5.0, 1e-10);
    ASSERT_NEAR(matrix_get(t, 2, 0), 3.0, 1e-10);
    ASSERT_NEAR(matrix_get(t, 2, 1), 6.0, 1e-10);

    matrix_free(m);
    matrix_free(t);
}

TEST(test_matrix_identity) {
    Matrix *I = matrix_identity(3);
    ASSERT_NOT_NULL(I);
    ASSERT_EQ(I->rows, 3);
    ASSERT_EQ(I->cols, 3);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (i == j) {
                ASSERT_NEAR(matrix_get(I, i, j), 1.0, 1e-10);
            } else {
                ASSERT_NEAR(matrix_get(I, i, j), 0.0, 1e-10);
            }
        }
    }

    matrix_free(I);
}

TEST(test_identity_multiplication) {
    /* A * I = A */
    Matrix *A = matrix_alloc(2, 2);
    matrix_set(A, 0, 0, 1.0);
    matrix_set(A, 0, 1, 2.0);
    matrix_set(A, 1, 0, 3.0);
    matrix_set(A, 1, 1, 4.0);

    Matrix *I = matrix_identity(2);
    Matrix *result = matrix_matmul(A, I);

    ASSERT_NOT_NULL(result);
    ASSERT_NEAR(matrix_get(result, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR(matrix_get(result, 0, 1), 2.0, 1e-10);
    ASSERT_NEAR(matrix_get(result, 1, 0), 3.0, 1e-10);
    ASSERT_NEAR(matrix_get(result, 1, 1), 4.0, 1e-10);

    matrix_free(A);
    matrix_free(I);
    matrix_free(result);
}

TEST(test_transpose_of_transpose) {
    /* (A^T)^T = A */
    Matrix *A = matrix_alloc(2, 3);
    matrix_set(A, 0, 0, 1.0);
    matrix_set(A, 0, 1, 2.0);
    matrix_set(A, 0, 2, 3.0);
    matrix_set(A, 1, 0, 4.0);
    matrix_set(A, 1, 1, 5.0);
    matrix_set(A, 1, 2, 6.0);

    Matrix *At = matrix_transpose(A);
    Matrix *Att = matrix_transpose(At);

    ASSERT_NOT_NULL(Att);
    ASSERT_EQ(Att->rows, A->rows);
    ASSERT_EQ(Att->cols, A->cols);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            ASSERT_NEAR(matrix_get(Att, i, j), matrix_get(A, i, j), 1e-10);
        }
    }

    matrix_free(A);
    matrix_free(At);
    matrix_free(Att);
}

int main(void) {
    printf("Matrix Operations Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_matrix_alloc_free);
    RUN_TEST(test_matrix_get_set);
    RUN_TEST(test_matrix_copy);
    RUN_TEST(test_matrix_add);
    RUN_TEST(test_matrix_matmul);
    RUN_TEST(test_matrix_transpose);
    RUN_TEST(test_matrix_identity);
    RUN_TEST(test_identity_multiplication);
    RUN_TEST(test_transpose_of_transpose);

    TEST_SUMMARY();
}
