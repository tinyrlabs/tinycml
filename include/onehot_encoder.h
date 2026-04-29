/**
 * @file onehot_encoder.h
 * @brief One-hot encoding transformer for categorical integer features
 */

#ifndef ONEHOT_ENCODER_H
#define ONEHOT_ENCODER_H

#include "matrix.h"

/**
 * @brief One-hot encoder for integer-valued categorical features
 *
 * Fits on a Matrix X where each column represents a categorical feature
 * with integer values. Transforms into a one-hot representation.
 */
typedef struct {
    int n_features;          /**< Number of input features (columns) */
    int *n_categories;       /**< Number of unique categories per feature */
    int **categories_;       /**< Category values per feature (sorted) */
    int total_output_cols;   /**< Sum of n_categories across all features */

    int is_fitted;           /**< Whether encoder has been fitted */
} OneHotEncoder;

/**
 * @brief Create a new OneHotEncoder
 * @return Pointer to newly allocated encoder, or NULL on failure
 */
OneHotEncoder* onehot_encoder_create(void);

/**
 * @brief Fit the encoder on training data
 * @param encoder Encoder to fit
 * @param X Feature matrix with integer-valued entries
 * @return 0 on success, -1 on error
 */
int onehot_encoder_fit(OneHotEncoder *encoder, const Matrix *X);

/**
 * @brief Transform data using fitted encoder
 * @param encoder Fitted encoder
 * @param X Feature matrix with integer-valued entries
 * @return One-hot encoded matrix (n_samples × total_output_cols), or NULL on error
 */
Matrix* onehot_encoder_transform(const OneHotEncoder *encoder, const Matrix *X);

/**
 * @brief Free encoder memory
 * @param encoder Encoder to free
 */
void onehot_encoder_free(OneHotEncoder *encoder);

#endif /* ONEHOT_ENCODER_H */
