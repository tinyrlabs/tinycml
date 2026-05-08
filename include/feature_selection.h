/**
 * feature_selection.h - Feature Selection Utilities
 *
 * Provides:
 * - SelectKBest: Select top k features by score
 * - VarianceThreshold: Remove low-variance features
 * - f_classif / f_regression: ANOVA F-test scoring
 * - mutual_info_classif / mutual_info_regression: Mutual information
 */

#ifndef FEATURE_SELECTION_H
#define FEATURE_SELECTION_H

#include "matrix.h"
#include "estimator.h"

/**
 * Feature scoring functions
 */
typedef enum {
    SCORE_F_CLASSIF,         // ANOVA F-value for classification
    SCORE_F_REGRESSION,      // F-value for regression
    SCORE_MUTUAL_INFO_CLASSIF,    // Mutual information (classification)
    SCORE_MUTUAL_INFO_REGRESSION, // Mutual information (regression)
    SCORE_CHI2               // Chi-squared (for non-negative features)
} ScoreFunction;

/**
 * SelectKBest - Select features according to k highest scores
 */
typedef struct {
    Estimator base;

    // Configuration
    ScoreFunction score_func;
    int k;                   // Number of features to select

    // Fitted attributes
    double *scores_;         // Score for each feature
    double *pvalues_;        // p-value for each feature (if applicable)
    int *support_;           // Mask of selected features (1 = selected)
    int n_features_;
    int n_features_selected_;
} SelectKBest;

/**
 * VarianceThreshold - Remove features with variance below threshold
 */
typedef struct {
    Estimator base;

    // Configuration
    double threshold;

    // Fitted attributes
    double *variances_;
    int *support_;
    int n_features_;
    int n_features_selected_;
} VarianceThreshold;

/**
 * RFE - Recursive Feature Elimination
 */
typedef struct {
    Estimator base;

    // Configuration
    Estimator *estimator;    // Estimator with feature_importances or coef_
    int n_features_to_select;
    int step;                // Features to remove per iteration (1 = one at a time)

    // Fitted attributes
    int *support_;           // Mask of selected features
    int *ranking_;           // Feature ranking (1 = best)
    int n_features_;
} RFE;

/* ============================================
 * SelectKBest API
 * ============================================ */

/**
 * Create SelectKBest transformer
 *
 * @param score_func Scoring function to use
 * @param k Number of top features to select (0 = all)
 * @return SelectKBest instance
 */
SelectKBest* select_k_best_create(ScoreFunction score_func, int k);

/**
 * Fit SelectKBest
 */
Estimator* select_k_best_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform - reduce X to selected features
 */
Matrix* select_k_best_transform(const Estimator *self, const Matrix *X);

/**
 * Get feature scores
 */
const double* select_k_best_scores(const SelectKBest *skb);

/**
 * Get feature support mask
 */
const int* select_k_best_get_support(const SelectKBest *skb);

/**
 * Clone
 */
Estimator* select_k_best_clone(const Estimator *self);

/**
 * Free
 */
void select_k_best_free(Estimator *self);

/**
 * Print summary
 */
void select_k_best_print_summary(const Estimator *self);

/* ============================================
 * VarianceThreshold API
 * ============================================ */

/**
 * Create VarianceThreshold transformer
 *
 * @param threshold Features with variance <= threshold are removed
 * @return VarianceThreshold instance
 */
VarianceThreshold* variance_threshold_create(double threshold);

/**
 * Fit VarianceThreshold
 */
Estimator* variance_threshold_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform - remove low-variance features
 */
Matrix* variance_threshold_transform(const Estimator *self, const Matrix *X);

/**
 * Get variances
 */
const double* variance_threshold_variances(const VarianceThreshold *vt);

/**
 * Get support mask
 */
const int* variance_threshold_get_support(const VarianceThreshold *vt);

/**
 * Clone
 */
Estimator* variance_threshold_clone(const Estimator *self);

/**
 * Free
 */
void variance_threshold_free(Estimator *self);

/* ============================================
 * Scoring Functions (standalone)
 * ============================================ */

/**
 * Compute ANOVA F-value for classification
 * Returns F-statistic and p-values for each feature
 *
 * @param X Feature matrix (n_samples x n_features)
 * @param y Class labels (n_samples x 1)
 * @param f_values Output: F-statistic for each feature (caller allocates)
 * @param p_values Output: p-value for each feature (caller allocates, can be NULL)
 * @return 0 on success
 */
int f_classif(const Matrix *X, const Matrix *y, double *f_values, double *p_values);

/**
 * Compute F-value for regression (correlation-based)
 */
int f_regression(const Matrix *X, const Matrix *y, double *f_values, double *p_values);

/**
 * Compute chi-squared statistic (for non-negative features)
 */
int chi2(const Matrix *X, const Matrix *y, double *chi2_values, double *p_values);

/**
 * Compute mutual information for classification
 * Uses binning-based estimation
 */
int mutual_info_classif(const Matrix *X, const Matrix *y, double *mi_values, int n_bins);

/**
 * Compute mutual information for regression
 */
int mutual_info_regression(const Matrix *X, const Matrix *y, double *mi_values, int n_bins);

/* ============================================
 * Utility Functions
 * ============================================ */

/**
 * Get feature importances from a fitted estimator
 * Works with: LinearRegression, DecisionTree, RandomForest
 *
 * @param estimator Fitted estimator
 * @param importances Output array (caller allocates n_features)
 * @return 0 on success, -1 if not supported
 */
int get_feature_importances(const Estimator *estimator, double *importances);

#endif /* FEATURE_SELECTION_H */
