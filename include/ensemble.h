/**
 * ensemble.h - Ensemble methods
 *
 * Provides:
 * - Random Forest (Classifier and Regressor)
 * - Bagging
 */

#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "matrix.h"
#include "estimator.h"
#include "decision_tree.h"

/**
 * Random Forest Classifier
 */
typedef struct {
    Estimator base;

    // Configuration
    int n_estimators;           // Number of trees
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;           // Features to consider at each split (0 = sqrt(n_features))
    int bootstrap;              // Whether to use bootstrap samples
    unsigned int seed;

    // Trees
    DecisionTreeClassifier **trees;
    int n_classes;
    int n_features;

    // Out-of-bag score
    double oob_score_;
} RandomForestClassifier;

/**
 * Random Forest Regressor
 */
typedef struct {
    Estimator base;

    int n_estimators;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;
    int bootstrap;
    unsigned int seed;

    DecisionTreeRegressor **trees;
    int n_features;

    double oob_score_;
} RandomForestRegressor;

/* ============================================
 * Random Forest Classifier API
 * ============================================ */

/**
 * Create Random Forest Classifier
 *
 * @param n_estimators Number of trees (default: 100)
 * @return RandomForestClassifier instance
 */
RandomForestClassifier* random_forest_classifier_create(int n_estimators);

/**
 * Create with full configuration
 */
RandomForestClassifier* random_forest_classifier_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
);

/**
 * Fit classifier
 */
Estimator* random_forest_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict class labels
 */
Matrix* random_forest_classifier_predict(const Estimator *self, const Matrix *X);

/**
 * Predict class probabilities (averaged across trees)
 */
Matrix* random_forest_classifier_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score (accuracy)
 */
double random_forest_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Get feature importances
 */
Matrix* random_forest_classifier_feature_importances(const RandomForestClassifier *rf);

/**
 * Clone
 */
Estimator* random_forest_classifier_clone(const Estimator *self);

/**
 * Free
 */
void random_forest_classifier_free(Estimator *self);

/**
 * Print summary
 */
void random_forest_classifier_print_summary(const Estimator *self);

/* ============================================
 * Random Forest Regressor API
 * ============================================ */

/**
 * Create Random Forest Regressor
 */
RandomForestRegressor* random_forest_regressor_create(int n_estimators);

/**
 * Create with full configuration
 */
RandomForestRegressor* random_forest_regressor_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
);

/**
 * Fit regressor
 */
Estimator* random_forest_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict values
 */
Matrix* random_forest_regressor_predict(const Estimator *self, const Matrix *X);

/**
 * Score (R²)
 */
double random_forest_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* random_forest_regressor_clone(const Estimator *self);

/**
 * Free
 */
void random_forest_regressor_free(Estimator *self);

#endif /* ENSEMBLE_H */
