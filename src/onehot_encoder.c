/**
 * @file onehot_encoder.c
 * @brief One-hot encoding transformer implementation
 */

#include "onehot_encoder.h"
#include <stdlib.h>
#include <string.h>

/* ============================================
 * Public API
 * ============================================ */

OneHotEncoder* onehot_encoder_create(void) {
    OneHotEncoder *enc = calloc(1, sizeof(OneHotEncoder));
    if (!enc) return NULL;

    enc->n_features       = 0;
    enc->n_categories     = NULL;
    enc->categories_      = NULL;
    enc->total_output_cols = 0;
    enc->is_fitted        = 0;

    return enc;
}

/* ============================================
 * Internal helpers
 * ============================================ */

static void encoder_free_params(OneHotEncoder *enc) {
    if (!enc) return;
    if (enc->n_categories) {
        free(enc->n_categories);
        enc->n_categories = NULL;
    }
    if (enc->categories_) {
        for (int j = 0; j < enc->n_features; j++) {
            free(enc->categories_[j]);
        }
        free(enc->categories_);
        enc->categories_ = NULL;
    }
    enc->n_features = 0;
    enc->total_output_cols = 0;
    enc->is_fitted = 0;
}

/**
 * Collect unique integer values from a column and sort them.
 * Returns the count of unique values and fills *out_categories.
 */
static int collect_unique_categories(const Matrix *X, size_t col,
                                      int **out_categories, int *out_count) {
    size_t N = X->rows;
    int *values = malloc(N * sizeof(int));
    if (!values) return -1;

    /* Collect all values */
    for (size_t i = 0; i < N; i++) {
        values[i] = (int)X->data[i * X->cols + col];
    }

    /* Find unique values (simple O(n^2) approach) */
    int *unique = malloc(N * sizeof(int));
    if (!unique) {
        free(values);
        return -1;
    }
    int n_unique = 0;

    for (size_t i = 0; i < N; i++) {
        int found = 0;
        for (int k = 0; k < n_unique; k++) {
            if (unique[k] == values[i]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            unique[n_unique++] = values[i];
        }
    }

    /* Sort unique values (simple insertion sort — small arrays) */
    for (int i = 1; i < n_unique; i++) {
        int key = unique[i];
        int j = i - 1;
        while (j >= 0 && unique[j] > key) {
            unique[j + 1] = unique[j];
            j--;
        }
        unique[j + 1] = key;
    }

    /* Allocate exact-sized output */
    *out_categories = malloc(n_unique * sizeof(int));
    if (!*out_categories) {
        free(values);
        free(unique);
        return -1;
    }
    memcpy(*out_categories, unique, n_unique * sizeof(int));
    *out_count = n_unique;

    free(values);
    free(unique);
    return 0;
}

/* ============================================
 * fit
 * ============================================ */

int onehot_encoder_fit(OneHotEncoder *encoder, const Matrix *X) {
    if (!encoder || !X) return -1;

    /* Free previous state */
    encoder_free_params(encoder);

    int F = (int)X->cols;
    encoder->n_features = F;
    encoder->n_categories = calloc(F, sizeof(int));
    encoder->categories_  = calloc(F, sizeof(int*));

    if (!encoder->n_categories || !encoder->categories_) {
        encoder_free_params(encoder);
        return -1;
    }

    encoder->total_output_cols = 0;

    for (int j = 0; j < F; j++) {
        if (collect_unique_categories(X, (size_t)j,
                                       &encoder->categories_[j],
                                       &encoder->n_categories[j]) != 0) {
            encoder_free_params(encoder);
            return -1;
        }
        encoder->total_output_cols += encoder->n_categories[j];
    }

    encoder->is_fitted = 1;
    return 0;
}

/* ============================================
 * transform
 * ============================================ */

Matrix* onehot_encoder_transform(const OneHotEncoder *encoder, const Matrix *X) {
    if (!encoder || !X || !encoder->is_fitted) return NULL;
    if ((int)X->cols != encoder->n_features) return NULL;

    size_t N = X->rows;
    int total_cols = encoder->total_output_cols;

    Matrix *result = matrix_alloc(N, (size_t)total_cols);
    if (!result) return NULL;

    /* Initialize all to 0.0 (matrix_alloc uses calloc-style zero init via its implementation) */
    /* matrix_alloc may or may not zero-init — explicitly fill */
    for (size_t i = 0; i < N * (size_t)total_cols; i++) {
        result->data[i] = 0.0;
    }

    for (size_t i = 0; i < N; i++) {
        int col_offset = 0;
        for (int j = 0; j < encoder->n_features; j++) {
            int val = (int)X->data[i * X->cols + j];
            int n_cats = encoder->n_categories[j];

            /* Find the index of this value in the category list */
            int found_idx = -1;
            for (int k = 0; k < n_cats; k++) {
                if (encoder->categories_[j][k] == val) {
                    found_idx = k;
                    break;
                }
            }

            if (found_idx >= 0) {
                result->data[i * (size_t)total_cols + col_offset + found_idx] = 1.0;
            }

            col_offset += n_cats;
        }
    }

    return result;
}

/* ============================================
 * free
 * ============================================ */

void onehot_encoder_free(OneHotEncoder *encoder) {
    if (encoder) {
        encoder_free_params(encoder);
        free(encoder);
    }
}
