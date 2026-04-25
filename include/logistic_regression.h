/**
 * @file logistic_regression.h
 * @brief Logistic regression for binary classification
 */

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"
#include "estimator.h"

/**
 * @brief Sigmoid function
 * @param x Input value
 * @return 1 / (1 + exp(-x))
 */
double sigmoid(double x);

/* ============================================
 * Legacy API (backward compatible)
 * ============================================ */

/**
 * @brief Fit logistic regression using gradient descent
 * @param X Feature matrix (n x m), should include bias column
 * @param y Binary target vector (n x 1), values 0 or 1
 * @param lr Learning rate
 * @param epochs Number of iterations
 * @return Weight vector (m x 1), or NULL on error
 */
Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict probabilities
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @return Probability vector (n x 1), or NULL on error
 */
Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights);

/**
 * @brief Predict class labels
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @param threshold Classification threshold (usually 0.5)
 * @return Class labels (n x 1), values 0 or 1, or NULL on error
 */
Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold);

/* ============================================
 * Estimator-compatible API
 * ============================================ */

/**
 * @brief Logistic Regression model with Estimator vtable
 *
 * Embeds Estimator base so it can be used with Pipeline,
 * GridSearchCV, cross_val_score, etc.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double learning_rate;   /**< Learning rate for gradient descent (default 0.1) */
    int max_iter;           /**< Maximum number of iterations (default 1000) */
    double lambda_val;      /**< L2 regularization strength (default 0.0) */
    double tol;             /**< Convergence tolerance (default 1e-6) */

    /* Fitted parameters (set after fit) */
    Matrix *weights;        /**< Learned weight vector (n_features x 1) */
    int n_features;         /**< Number of features */
    int n_classes;          /**< Number of classes (2 for binary) */
} LogisticRegressionModel;

/**
 * @brief Create a new LogisticRegressionModel with default hyperparameters
 * @return Pointer to newly allocated model, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
LogisticRegressionModel* logreg_model_create(void);

/**
 * @brief Save fitted LogisticRegressionModel to binary file
 * @return 0 on success, -1 on error
 */
int logreg_model_save(const Estimator *self, const char *path);

/**
 * @brief Load LogisticRegressionModel from binary file
 * @return New fitted model, or NULL on error
 */
Estimator* logreg_model_load(const char *path);

/* ============================================
 * Softmax (Multi-class) Logistic Regression
 * ============================================ */

/**
 * @brief Softmax Regression model for multi-class classification
 *
 * Uses the softmax function to generalize logistic regression to multiple classes.
 * Embeds Estimator base so it can be used with Pipeline, cross_val_score, etc.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double learning_rate;   /**< Learning rate for gradient descent (default 0.1) */
    int max_iter;           /**< Maximum number of iterations (default 1000) */
    double tol;             /**< Convergence tolerance (default 1e-6) */

    /* Fitted parameters (set after fit) */
    Matrix *weights;        /**< Learned weight matrix (n_classes × n_features) */
    double *biases;         /**< Bias terms per class (n_classes) */
    int n_features;         /**< Number of features */
    int n_classes;          /**< Number of classes */
} SoftmaxRegressionModel;

/**
 * @brief Create a new SoftmaxRegressionModel with default hyperparameters
 * @return Pointer to newly allocated model, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
SoftmaxRegressionModel* softmax_model_create(void);

/**
 * @brief Save fitted SoftmaxRegressionModel to binary file
 * @return 0 on success, -1 on error
 */
int softmax_model_save(const Estimator *self, const char *path);

/**
 * @brief Load SoftmaxRegressionModel from binary file
 * @return New fitted model, or NULL on error
 */
Estimator* softmax_model_load(const char *path);

#endif /* LOGISTIC_REGRESSION_H */
