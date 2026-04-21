/**
 * @file test_harness.h
 * @brief Simple test framework for ML library
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) void name(void)

#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    fflush(stdout); \
    tests_run++; \
    name(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED\n  Assertion failed: %s\n  at %s:%d\n", \
               #cond, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))

#define ASSERT_NEAR(a, b, eps) do { \
    double _a = (a); \
    double _b = (b); \
    double _eps = (eps); \
    if (fabs(_a - _b) >= _eps) { \
        printf("FAILED\n  ASSERT_NEAR failed: %f != %f (eps=%f)\n  at %s:%d\n", \
               _a, _b, _eps, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) ASSERT((ptr) != NULL)

#define ASSERT_NULL(ptr) ASSERT((ptr) == NULL)

#define TEST_SUMMARY() do { \
    printf("\n========================================\n"); \
    printf("%d/%d tests passed\n", tests_passed, tests_run); \
    printf("========================================\n"); \
    return tests_passed == tests_run ? 0 : 1; \
} while(0)

#endif /* TEST_HARNESS_H */
