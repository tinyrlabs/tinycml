/**
 * @file metrics.h
 * @brief Evaluation metrics for ML models
 */

#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"

/**
 * @brief Confusion matrix structure
 */
typedef struct {
    int tp;  /**< True Positives */
    int tn;  /**< True Negatives */
    int fp;  /**< False Positives */
    int fn;  /**< False Negatives */
} ConfusionMatrix;

/* Regression Metrics */

/**
 * @brief Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MSE value
 */
double mse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Root Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return RMSE value
 */
double rmse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Mean Absolute Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MAE value
 */
double mae(const Matrix *y_true, const Matrix *y_pred);

/* Classification Metrics */

/**
 * @brief Classification accuracy
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @return Accuracy (0 to 1)
 */
double accuracy(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Precision (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Precision value
 */
double precision(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Recall (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Recall value
 */
double recall(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief F1 Score (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return F1 score
 */
double f1_score(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Compute confusion matrix
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Confusion matrix
 */
ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Print confusion matrix
 * @param cm Confusion matrix to print
 */
void confusion_matrix_print(const ConfusionMatrix *cm);

/* Multi-class metrics */

/**
 * @brief Multi-class confusion matrix: returns n_classes x n_classes matrix
 * @param y_true True labels (integer class labels)
 * @param y_pred Predicted labels (integer class labels)
 * @return Dynamically allocated n_classes x n_classes Matrix (caller must free), or NULL on error
 *
 * Labels are assumed to be integers 0..n_classes-1. The matrix is auto-sized
 * to (max_label+1) x (max_label+1). Entry [i][j] = count of samples with
 * true class i predicted as class j.
 */
Matrix* confusion_matrix_multi(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Macro-averaged precision across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged precision
 */
double precision_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Macro-averaged recall across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged recall
 */
double recall_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Macro-averaged F1 score across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged F1 score
 */
double f1_score_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Weighted-averaged F1 score (weighted by class support)
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Weighted F1 score
 */
double f1_score_weighted(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/* Clustering Metrics */

/**
 * @brief Compute mean Silhouette Coefficient for clustering
 * @param X Data matrix (n_samples x n_features)
 * @param labels Cluster labels (n_samples x 1), integer labels 0..k-1
 * @param n_clusters Number of clusters
 * @return Mean silhouette score in [-1, 1], or 0.0 on error
 *
 * For each sample i:
 *   a(i) = mean distance to other samples in same cluster
 *   b(i) = min over other clusters of mean distance to samples in that cluster
 *   s(i) = (b(i) - a(i)) / max(a(i), b(i))
 * Returns mean of s(i) over all samples.
 * Uses Euclidean distance.
 */
double silhouette_score(const Matrix *X, const Matrix *labels, int n_clusters);

#endif /* METRICS_H */
