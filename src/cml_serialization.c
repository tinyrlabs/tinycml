/**
 * cml_serialization.c - Shared serialization helpers
 */

#include "cml_serialization.h"
#include <string.h>

static const char cml_ser_magic[4] = {'C', 'M', 'L', '\0'};

int cml_ser_write_header(FILE *f, int subtype) {
    if (fwrite(cml_ser_magic, 1, CML_SER_MAGIC_SIZE, f) != CML_SER_MAGIC_SIZE) return -1;
    if (fwrite(&subtype, sizeof(int), 1, f) != 1) return -1;
    return 0;
}

int cml_ser_check_header(FILE *f, int expected_subtype) {
    char magic[4];
    if (fread(magic, 1, CML_SER_MAGIC_SIZE, f) != CML_SER_MAGIC_SIZE) return -1;
    if (memcmp(magic, cml_ser_magic, 4) != 0) return -1;
    int subtype;
    if (fread(&subtype, sizeof(int), 1, f) != 1) return -1;
    if (subtype != expected_subtype) return -1;
    return 0;
}

int cml_ser_write_matrix(FILE *f, const Matrix *m) {
    int rows = (int)m->rows;
    int cols = (int)m->cols;
    if (fwrite(&rows, sizeof(int), 1, f) != 1) return -1;
    if (fwrite(&cols, sizeof(int), 1, f) != 1) return -1;
    size_t total = m->rows * m->cols;
    if (fwrite(m->data, sizeof(double), total, f) != total) return -1;
    return 0;
}

Matrix* cml_ser_read_matrix(FILE *f) {
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, f) != 1) return NULL;
    if (fread(&cols, sizeof(int), 1, f) != 1) return NULL;
    Matrix *m = matrix_alloc((size_t)rows, (size_t)cols);
    if (!m) return NULL;
    size_t total = (size_t)rows * (size_t)cols;
    if (fread(m->data, sizeof(double), total, f) != total) {
        matrix_free(m);
        return NULL;
    }
    return m;
}

int cml_ser_write_doubles(FILE *f, const double *data, int n) {
    return fwrite(data, sizeof(double), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_read_doubles(FILE *f, double *data, int n) {
    return fread(data, sizeof(double), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_write_ints(FILE *f, const int *data, int n) {
    return fwrite(data, sizeof(int), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_read_ints(FILE *f, int *data, int n) {
    return fread(data, sizeof(int), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_write_double(FILE *f, double val) {
    return fwrite(&val, sizeof(double), 1, f) == 1 ? 0 : -1;
}

int cml_ser_read_double(FILE *f, double *val) {
    return fread(val, sizeof(double), 1, f) == 1 ? 0 : -1;
}

int cml_ser_write_int(FILE *f, int val) {
    return fwrite(&val, sizeof(int), 1, f) == 1 ? 0 : -1;
}

int cml_ser_read_int(FILE *f, int *val) {
    return fread(val, sizeof(int), 1, f) == 1 ? 0 : -1;
}
