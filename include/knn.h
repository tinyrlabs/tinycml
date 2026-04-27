/**
 * @file knn.h
 * @brief k-Nearest Neighbors classifier
 */

#ifndef KNN_H
#define KNN_H

#include "matrix.h"

/**
 * @brief k-NN model structure
 */
typedef struct {
    Matrix *X_train;  /**< Training features */
    Matrix *y_train;  /**< Training labels */
    int k;            /**< Number of neighbors */
} KNNModel;

/**
 * @brief Fit k-NN model (stores training data)
 * @param X Training feature matrix
 * @param y Training labels (class indices as doubles)
 * @param k Number of neighbors
 * @return Fitted model, or NULL on error
 */
KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k);

/**
 * @brief Predict class labels for new samples
 * @param model Fitted k-NN model
 * @param X Feature matrix to predict
 * @return Predicted class labels, or NULL on error
 */
Matrix* knn_predict(const KNNModel *model, const Matrix *X);

/**
 * @brief Predict class probabilities for new samples
 * @param model Fitted k-NN model
 * @param X Feature matrix to predict
 * @param n_classes Number of classes
 * @return n_samples x n_classes probability matrix, or NULL on error
 */
Matrix* knn_predict_proba(const KNNModel *model, const Matrix *X, int n_classes);

/**
 * @brief Free k-NN model
 * @param model Model to free
 */
void knn_free(KNNModel *model);

#endif /* KNN_H */
