/**
 * @file test_cml_error.c
 * @brief Tests for the unified error handling system
 */

#include "cml_error.h"
#include "matrix.h"
#include "test_harness.h"

TEST(test_cml_set_get) {
    cml_set_error(CML_ERROR_MEMORY, "test alloc failed");
    CMLError err = cml_get_last_error();
    ASSERT(err.code == CML_ERROR_MEMORY);
    ASSERT(strstr(err.message, "test alloc failed") != NULL);
}

TEST(test_cml_null_check) {
    /* matrix_matmul should set error on NULL input */
    Matrix *m = NULL;
    Matrix *result = matrix_matmul(m, m);
    ASSERT(result == NULL);
    CMLError err = cml_get_last_error();
    ASSERT(err.code != CML_OK);
}

TEST(test_cml_status_string) {
    ASSERT(strcmp(cml_status_string(CML_OK), "OK") == 0);
    ASSERT(strcmp(cml_status_string(CML_ERROR_MEMORY), "Memory allocation failed") == 0);
    ASSERT(strcmp(cml_status_string(CML_ERROR_NULL_PTR), "NULL pointer") == 0);
    ASSERT(strcmp(cml_status_string(CML_ERROR_OUT_OF_BOUNDS), "Out of bounds") == 0);
    ASSERT(strcmp(cml_status_string(CML_ERROR_NOT_FITTED), "Model not fitted") == 0);
}

TEST(test_cml_matrix_error) {
    Matrix *m = matrix_alloc(3, 3);
    ASSERT_NOT_NULL(m);
    /* Trigger out-of-bounds */
    double val = matrix_get(m, 10, 10);
    (void)val;
    CMLError err = cml_get_last_error();
    ASSERT(err.code == CML_ERROR_OUT_OF_BOUNDS);
    matrix_free(m);
}

int main(void) {
    RUN_TEST(test_cml_set_get);
    RUN_TEST(test_cml_null_check);
    RUN_TEST(test_cml_status_string);
    RUN_TEST(test_cml_matrix_error);

    TEST_SUMMARY();
}
