/**
 * cml_serialization.h - Shared serialization helpers for tinycml models
 *
 * Provides common binary format utilities for save/load:
 *   MAGIC (4 bytes: "CML\0")
 *   SUBTYPE (4 bytes: int)
 *   Model-specific data
 */

#ifndef CML_SERIALIZATION_H
#define CML_SERIALIZATION_H

#include <stdio.h>
#include "matrix.h"

/* Magic bytes: "CML\0" */
#define CML_SER_MAGIC_SIZE 4

/* Model subtype IDs for distinguishing models sharing a ModelType */
enum {
    CML_SUB_LOGREG_BINARY     = 1,
    CML_SUB_SOFTMAX           = 2,
    CML_SUB_GAUSSIAN_NB       = 3,
    CML_SUB_MULTINOMIAL_NB    = 4,
    CML_SUB_LINEAR_SVC        = 5,
    CML_SUB_SVM_CLASSIFIER    = 6,
    CML_SUB_RIDGE             = 7,
    CML_SUB_LASSO             = 8,
    CML_SUB_DECISION_TREE_CLF = 9,
};

/** Write serialization header (magic + subtype). Returns 0 on success. */
int cml_ser_write_header(FILE *f, int subtype);

/** Read and verify serialization header. Returns 0 on success. */
int cml_ser_check_header(FILE *f, int expected_subtype);

/** Write a matrix in serialization format (rows, cols, data). */
int cml_ser_write_matrix(FILE *f, const Matrix *m);

/** Read a matrix from serialization format. Returns new matrix or NULL. */
Matrix* cml_ser_read_matrix(FILE *f);

/** Write an array of n doubles. */
int cml_ser_write_doubles(FILE *f, const double *data, int n);

/** Read an array of n doubles. */
int cml_ser_read_doubles(FILE *f, double *data, int n);

/** Write an array of n ints. */
int cml_ser_write_ints(FILE *f, const int *data, int n);

/** Read an array of n ints. */
int cml_ser_read_ints(FILE *f, int *data, int n);

/** Write a single double. */
int cml_ser_write_double(FILE *f, double val);

/** Read a single double. */
int cml_ser_read_double(FILE *f, double *val);

/** Write a single int. */
int cml_ser_write_int(FILE *f, int val);

/** Read a single int. */
int cml_ser_read_int(FILE *f, int *val);

#endif /* CML_SERIALIZATION_H */
