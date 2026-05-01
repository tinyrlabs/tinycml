/**
 * @file svm.h
 * @brief Linear Support Vector Machine classifier
 */

#ifndef SVM_H
#define SVM_H

#include "matrix.h"
#include "estimator.h"

/**
 * @brief Linear SVM classifier using sub-gradient descent with hinge loss
 *
 * Supports binary classification with labels {-1, +1}.
 * If input labels are {0, 1}, they are converted internally.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double C;              /**< Regularization parameter (default 1.0) */
    double learning_rate;  /**< Learning rate for gradient descent (default 0.001) */
    int max_iter;          /**< Maximum number of epochs (default 1000) */
    double tol;            /**< Convergence tolerance (default 1e-4) */

    /* Fitted parameters */
    Matrix *weights;       /**< Weight vector (n_features x 1) */
    double bias;           /**< Bias term */
    int n_features;        /**< Number of features */

    /* Internal */
    int labels_converted;  /**< Whether input labels were converted from {0,1} to {-1,+1} */
} LinearSVC;

/**
 * @brief Create a new Linear SVM classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 */
LinearSVC* linear_svc_create(void);

/**
 * @brief Free a Linear SVM classifier
 * @param svc Classifier to free (passed as Estimator* for vtable compatibility)
 */
void linear_svc_free_impl(Estimator *self);

/**
 * @brief Save a fitted LinearSVC to a binary file
 * @param self Fitted Estimator pointer
 * @param path File path
 * @return 0 on success, -1 on error
 */
int linear_svc_save(const Estimator *self, const char *path);

/**
 * @brief Load a LinearSVC from a binary file
 * @param path File path
 * @return New fitted Estimator, or NULL on error
 */
Estimator* linear_svc_load(const char *path);

/**
 * @brief Predict class probabilities for new samples
 * @param model Fitted LinearSVC model
 * @param X Feature matrix
 * @return n_samples x 1 matrix of probabilities P(y=1), or NULL on error
 */
Matrix* linear_svc_predict_proba(const LinearSVC *model, const Matrix *X);

/* ============================================
 * SVM with kernel support (Linear + RBF)
 * ============================================ */

typedef enum {
    CML_KERNEL_LINEAR = 0,
    CML_KERNEL_RBF = 1
} SVMKernelType;

typedef struct {
    Estimator base;
    SVMKernelType kernel;
    double C;           /**< Regularization (default 1.0) */
    double gamma;       /**< RBF gamma (default 1.0/n_features, -1 = auto) */
    double lr;          /**< Learning rate for SGD */
    int max_iter;
    double tol;

    /* Fitted (linear kernel) */
    Matrix *weights;
    double bias;

    /* Fitted (kernel) */
    Matrix *support_vectors;  /**< n_support x n_features */
    double *alphas;           /**< Dual coefficients (stored with y_i baked in) */
    double *labels_train;     /**< Training labels y_i in {-1, +1} */
    int n_support;
    int n_features;

    int labels_converted;     /**< Whether input labels were converted */
} SVMClassifier;

/**
 * @brief Create a new SVM classifier with specified kernel
 * @param kernel Kernel type (CML_KERNEL_LINEAR or CML_KERNEL_RBF)
 * @return Pointer to newly allocated classifier, or NULL on failure
 */
SVMClassifier* svm_classifier_create(SVMKernelType kernel);

/**
 * @brief Free an SVMClassifier
 * @param self Estimator pointer to free
 */
void svm_classifier_free_impl(Estimator *self);

/**
 * @brief Save a fitted SVMClassifier to a binary file
 * @param self Fitted Estimator pointer
 * @param path File path
 * @return 0 on success, -1 on error
 */
int svm_classifier_save(const Estimator *self, const char *path);

/**
 * @brief Load an SVMClassifier from a binary file
 * @param path File path
 * @return New fitted Estimator, or NULL on error
 */
Estimator* svm_classifier_load(const char *path);

#endif /* SVM_H */
