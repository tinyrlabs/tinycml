/**
 * @file naive_bayes.h
 * @brief Gaussian Naive Bayes classifier
 */

#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "matrix.h"
#include "estimator.h"

/**
 * @brief Gaussian Naive Bayes classifier
 *
 * Assumes features are conditionally independent given the class
 * and follow a Gaussian distribution.
 */
typedef struct {
    Estimator base;

    /* Training parameters */
    double var_smoothing;   /**< Variance smoothing added to variances (default 1e-9) */

    /* Fitted parameters (set after fit) */
    int n_classes;          /**< Number of unique classes */
    int n_features;         /**< Number of features */
    double *class_priors;       /**< P(class), size n_classes */
    int *class_count_;          /**< Sample count per class */
    double *class_log_prior_;   /**< log(P(class)) */
    double *theta_;             /**< Mean per (class, feature), n_classes × n_features row-major */
    double *var_;               /**< Variance per (class, feature), n_classes × n_features row-major */

    int total_samples_;         /**< Total number of training samples */
} GaussianNaiveBayes;

/**
 * @brief Create a new Gaussian Naive Bayes classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
GaussianNaiveBayes* gaussian_nb_create(void);

/**
 * @brief Save fitted GaussianNaiveBayes to binary file
 */
int gaussian_nb_save(const Estimator *self, const char *path);

/**
 * @brief Load GaussianNaiveBayes from binary file
 */
Estimator* gaussian_nb_load(const char *path);

/**
 * @brief Compute class-conditional probabilities for each sample
 * @param self Estimator pointer (must be fitted)
 * @param X Feature matrix (n_samples × n_features)
 * @return Matrix of probabilities (n_samples × n_classes), or NULL on error
 */
Matrix* gaussian_nb_predict_proba(const Estimator *self, const Matrix *X);

/* ============================================
 * Multinomial Naive Bayes
 * ============================================ */

/**
 * @brief Multinomial Naive Bayes classifier
 *
 * Suitable for count-based features (e.g., word counts in text classification).
 * Uses Laplace smoothing to avoid zero probabilities.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double alpha;           /**< Laplace smoothing parameter (default 1.0) */

    /* Fitted parameters (set after fit) */
    Matrix *log_prior;      /**< Log prior probabilities (n_classes × 1) */
    Matrix *theta;          /**< Log-likelihood per (class, feature) (n_classes × n_features) */
    int n_classes;          /**< Number of unique classes */
    int n_features;         /**< Number of features */
} MultinomialNB;

/**
 * @brief Create a new MultinomialNB classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
MultinomialNB* multinomial_nb_create(void);

/**
 * @brief Save fitted MultinomialNB to binary file
 */
int multinomial_nb_save(const Estimator *self, const char *path);

/**
 * @brief Load MultinomialNB from binary file
 */
Estimator* multinomial_nb_load(const char *path);

#endif /* NAIVE_BAYES_H */
