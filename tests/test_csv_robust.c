/**
 * @file test_csv_robust.c
 * @brief Tests for the robust CSV parser
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "csv.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_FEQ(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (fabs(_a - _b) > _t) { \
        printf("  FAIL: %s: expected %.6f, got %.6f (line %d)\n", msg, _b, _a, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    if (f) { fputs(content, f); fclose(f); }
}

/* ── Test 1: Basic CSV ──────────────────────────────────────────────── */

static void test_csv_basic(void) {
    printf("test_csv_basic...\n");
    const char *path = "/tmp/test_csv_basic.csv";
    write_file(path, "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n");

    CSVOptions opts = {
        .has_header = 0,
        .delimiter = ',',
        .quote = '"',
        .missing_val = NAN,
        .skip_empty_lines = 1,
        .max_rows = 0,
        .max_cols = 0
    };

    Matrix *m = csv_load_ext(path, &opts);
    ASSERT(m != NULL, "csv_load_ext returned NULL");

    ASSERT_FEQ(matrix_get(m, 0, 0), 1.0, 1e-9, "m[0][0]");
    ASSERT_FEQ(matrix_get(m, 0, 1), 2.0, 1e-9, "m[0][1]");
    ASSERT_FEQ(matrix_get(m, 0, 2), 3.0, 1e-9, "m[0][2]");
    ASSERT_FEQ(matrix_get(m, 1, 0), 4.0, 1e-9, "m[1][0]");
    ASSERT_FEQ(matrix_get(m, 2, 2), 9.0, 1e-9, "m[2][2]");

    ASSERT(m->rows == 3, "row count");
    ASSERT(m->cols == 3, "col count");

    matrix_free(m);
    unlink(path);
    tests_passed++;
    printf("  PASS\n");
}

/* ── Test 2: Quoted fields ──────────────────────────────────────────── */

static void test_csv_quoted(void) {
    printf("test_csv_quoted...\n");
    const char *path = "/tmp/test_csv_quoted.csv";
    /* "hello, world" should parse as one field; we test numeric value
       is NaN for non-numeric, and verify the numeric fields around it. */
    write_file(path, "1.0,\"hello, world\",3.0\n4.0,5.5,6.0\n");

    CSVOptions opts = {
        .has_header = 0,
        .delimiter = ',',
        .quote = '"',
        .missing_val = NAN,
        .skip_empty_lines = 1,
        .max_rows = 0,
        .max_cols = 0
    };

    Matrix *m = csv_load_ext(path, &opts);
    ASSERT(m != NULL, "csv_load_ext returned NULL");

    ASSERT_FEQ(matrix_get(m, 0, 0), 1.0, 1e-9, "m[0][0]");
    ASSERT(isnan(matrix_get(m, 0, 1)), "m[0][1] should be NaN (non-numeric quoted)");
    ASSERT_FEQ(matrix_get(m, 0, 2), 3.0, 1e-9, "m[0][2]");
    ASSERT_FEQ(matrix_get(m, 1, 0), 4.0, 1e-9, "m[1][0]");
    ASSERT_FEQ(matrix_get(m, 1, 1), 5.5, 1e-9, "m[1][1]");
    ASSERT_FEQ(matrix_get(m, 1, 2), 6.0, 1e-9, "m[1][2]");

    matrix_free(m);
    unlink(path);
    tests_passed++;
    printf("  PASS\n");
}

/* ── Test 3: Missing values → NAN ───────────────────────────────────── */

static void test_csv_missing(void) {
    printf("test_csv_missing...\n");
    const char *path = "/tmp/test_csv_missing.csv";
    write_file(path, "1.0,,3.0\n,5.0,\n7.0,8.0,9.0\n");

    CSVOptions opts = {
        .has_header = 0,
        .delimiter = ',',
        .quote = '"',
        .missing_val = NAN,
        .skip_empty_lines = 1,
        .max_rows = 0,
        .max_cols = 0
    };

    Matrix *m = csv_load_ext(path, &opts);
    ASSERT(m != NULL, "csv_load_ext returned NULL");

    ASSERT_FEQ(matrix_get(m, 0, 0), 1.0, 1e-9, "m[0][0]");
    ASSERT(isnan(matrix_get(m, 0, 1)), "m[0][1] should be NaN (empty)");
    ASSERT_FEQ(matrix_get(m, 0, 2), 3.0, 1e-9, "m[0][2]");

    ASSERT(isnan(matrix_get(m, 1, 0)), "m[1][0] should be NaN (empty)");
    ASSERT_FEQ(matrix_get(m, 1, 1), 5.0, 1e-9, "m[1][1]");
    ASSERT(isnan(matrix_get(m, 1, 2)), "m[1][2] should be NaN (empty)");

    ASSERT_FEQ(matrix_get(m, 2, 0), 7.0, 1e-9, "m[2][0]");
    ASSERT_FEQ(matrix_get(m, 2, 1), 8.0, 1e-9, "m[2][1]");
    ASSERT_FEQ(matrix_get(m, 2, 2), 9.0, 1e-9, "m[2][2]");

    matrix_free(m);
    unlink(path);
    tests_passed++;
    printf("  PASS\n");
}

/* ── Test 4: Tab delimiter ──────────────────────────────────────────── */

static void test_csv_delimiter(void) {
    printf("test_csv_delimiter...\n");
    const char *path = "/tmp/test_csv_delimiter.csv";
    write_file(path, "1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n");

    CSVOptions opts = {
        .has_header = 0,
        .delimiter = '\t',
        .quote = '"',
        .missing_val = NAN,
        .skip_empty_lines = 1,
        .max_rows = 0,
        .max_cols = 0
    };

    Matrix *m = csv_load_ext(path, &opts);
    ASSERT(m != NULL, "csv_load_ext returned NULL");

    ASSERT_FEQ(matrix_get(m, 0, 0), 1.0, 1e-9, "m[0][0]");
    ASSERT_FEQ(matrix_get(m, 0, 1), 2.0, 1e-9, "m[0][1]");
    ASSERT_FEQ(matrix_get(m, 0, 2), 3.0, 1e-9, "m[0][2]");
    ASSERT_FEQ(matrix_get(m, 1, 0), 4.0, 1e-9, "m[1][0]");
    ASSERT_FEQ(matrix_get(m, 1, 1), 5.0, 1e-9, "m[1][1]");
    ASSERT_FEQ(matrix_get(m, 1, 2), 6.0, 1e-9, "m[1][2]");

    ASSERT(m->rows == 2, "row count");
    ASSERT(m->cols == 3, "col count");

    matrix_free(m);
    unlink(path);
    tests_passed++;
    printf("  PASS\n");
}

/* ── Test 5: Auto-detect delimiter ──────────────────────────────────── */

static void test_csv_detect(void) {
    printf("test_csv_detect...\n");

    /* Test tab detection */
    const char *path_tab = "/tmp/test_csv_detect_tab.csv";
    write_file(path_tab, "1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n");
    char d = csv_detect_delimiter(path_tab);
    ASSERT(d == '\t', "tab should be detected");
    unlink(path_tab);

    /* Test semicolon detection */
    const char *path_semi = "/tmp/test_csv_detect_semi.csv";
    write_file(path_semi, "1.0;2.0;3.0\n4.0;5.0;6.0\n");
    d = csv_detect_delimiter(path_semi);
    ASSERT(d == ';', "semicolon should be detected");
    unlink(path_semi);

    /* Test comma detection (default) */
    const char *path_comma = "/tmp/test_csv_detect_comma.csv";
    write_file(path_comma, "1.0,2.0,3.0\n4.0,5.0,6.0\n");
    d = csv_detect_delimiter(path_comma);
    ASSERT(d == ',', "comma should be detected");
    unlink(path_comma);

    tests_passed++;
    printf("  PASS\n");
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== CSV Robust Parser Tests ===\n\n");

    test_csv_basic();
    test_csv_quoted();
    test_csv_missing();
    test_csv_delimiter();
    test_csv_detect();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
