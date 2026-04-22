/**
 * @file preprocessing.h
 * @brief Data preprocessing functions for ML library
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "matrix.h"

/**
 * @brief Result of train/test split
 */
typedef struct {
    Matrix *X_train;
    Matrix *X_test;
    Matrix *y_train;
    Matrix *y_test;
} TrainTestSplit;

/**
 * @brief Standard scaler (Z-score normalization)
 */
typedef struct {
    double *means;
    double *stds;
    size_t n_features;
} Scaler;

/**
 * @brief Min-max scaler
 */
typedef struct {
    double *mins;
    double *maxs;
    size_t n_features;
} MinMaxScaler;

/**
 * @brief Split data into training and test sets
 * @param X Feature matrix (n_samples x n_features)
 * @param y Target vector (n_samples x 1)
 * @param test_ratio Fraction for test set (e.g., 0.2)
 * @param seed Random seed for reproducibility
 * @return TrainTestSplit structure
 */
TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed);

/**
 * @brief Free memory in TrainTestSplit structure
 * @param split Split to free
 */
void train_test_split_free(TrainTestSplit *split);

/**
 * @brief Fit standard scaler to data
 * @param X Feature matrix
 * @return Fitted scaler, or NULL on error
 */
Scaler* standardize_fit(const Matrix *X);

/**
 * @brief Transform data using fitted scaler
 * @param X Feature matrix
 * @param scaler Fitted scaler
 * @return Transformed matrix, or NULL on error
 */
Matrix* standardize_transform(const Matrix *X, const Scaler *scaler);

/**
 * @brief Fit and transform in one step
 * @param X Feature matrix
 * @param scaler_out Output pointer for fitted scaler
 * @return Transformed matrix, or NULL on error
 */
Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out);

/**
 * @brief Free scaler memory
 * @param scaler Scaler to free
 */
void scaler_free(Scaler *scaler);

/**
 * @brief Fit min-max scaler to data
 * @param X Feature matrix
 * @return Fitted scaler, or NULL on error
 */
MinMaxScaler* minmax_fit(const Matrix *X);

/**
 * @brief Transform data using fitted min-max scaler
 * @param X Feature matrix
 * @param scaler Fitted scaler
 * @return Transformed matrix (scaled to [0,1]), or NULL on error
 */
Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler);

/**
 * @brief Free min-max scaler memory
 * @param scaler Scaler to free
 */
void minmax_scaler_free(MinMaxScaler *scaler);

/**
 * @brief Add bias column (ones) to matrix
 * @param X Feature matrix (n x m)
 * @return New matrix (n x m+1) with first column as ones
 */
Matrix* add_bias_column(const Matrix *X);

#endif /* PREPROCESSING_H */
