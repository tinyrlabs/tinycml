/**
 * validation.h - Cross-validation and model selection utilities
 *
 * Provides scikit-learn style cross-validation:
 * - k-fold cross-validation
 * - Stratified k-fold (for classification)
 * - Leave-one-out
 * - cross_val_score() function
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include "matrix.h"
#include "estimator.h"

/**
 * K-Fold split indices
 */
typedef struct {
    size_t *train_indices;
    size_t n_train;
    size_t *test_indices;
    size_t n_test;
} FoldIndices;

/**
 * K-Fold cross-validator
 */
typedef struct {
    int n_splits;           // Number of folds
    int shuffle;            // Whether to shuffle before splitting
    unsigned int seed;      // Random seed for shuffling
    size_t n_samples;       // Total number of samples
    size_t *indices;        // Shuffled indices
    int current_fold;       // Current fold being iterated
} KFold;

/**
 * Stratified K-Fold cross-validator
 */
typedef struct {
    int n_splits;
    int shuffle;
    unsigned int seed;
    size_t n_samples;
    size_t *indices;
    const Matrix *y;        // Target labels for stratification
    int n_classes;          // Number of unique classes
} StratifiedKFold;

/**
 * Cross-validation results
 */
typedef struct {
    double *test_scores;    // Score for each fold
    double *train_scores;   // Training score for each fold (optional)
    double *fit_times;      // Time to fit each fold
    double *score_times;    // Time to score each fold
    int n_splits;           // Number of folds
    double mean_test_score;
    double std_test_score;
    double mean_train_score;
    double std_train_score;
} CrossValResults;

/* ============================================
 * K-Fold Cross-Validation
 * ============================================ */

/**
 * Create a K-Fold cross-validator
 *
 * @param n_splits Number of folds (typically 5 or 10)
 * @param shuffle Whether to shuffle data before splitting
 * @param seed Random seed for shuffling
 * @return KFold structure
 */
KFold* kfold_create(int n_splits, int shuffle, unsigned int seed);

/**
 * Initialize K-Fold with data size
 */
void kfold_init(KFold *kf, size_t n_samples);

/**
 * Get fold indices for a specific fold
 */
FoldIndices kfold_get_fold(const KFold *kf, int fold);

/**
 * Free fold indices
 */
void fold_indices_free(FoldIndices *fi);

/**
 * Free K-Fold
 */
void kfold_free(KFold *kf);

/* ============================================
 * Stratified K-Fold (for classification)
 * ============================================ */

/**
 * Create a Stratified K-Fold cross-validator
 * Ensures each fold has approximately the same class distribution
 */
StratifiedKFold* stratified_kfold_create(int n_splits, int shuffle, unsigned int seed);

/**
 * Initialize with labels
 */
void stratified_kfold_init(StratifiedKFold *skf, const Matrix *y);

/**
 * Get fold indices
 */
FoldIndices stratified_kfold_get_fold(const StratifiedKFold *skf, int fold);

/**
 * Free Stratified K-Fold
 */
void stratified_kfold_free(StratifiedKFold *skf);

/* ============================================
 * Cross-validation scoring
 * ============================================ */

/**
 * Evaluate estimator using cross-validation
 *
 * Similar to sklearn.model_selection.cross_val_score
 *
 * @param estimator Model to evaluate (will be cloned for each fold)
 * @param X Feature matrix
 * @param y Target vector
 * @param n_splits Number of folds
 * @param shuffle Whether to shuffle data
 * @param seed Random seed
 * @return CrossValResults with scores for each fold
 */
CrossValResults* cross_val_score(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
);

/**
 * Cross-validation with stratified folds (for classification)
 */
CrossValResults* cross_val_score_stratified(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
);

/**
 * Cross-validation with custom scoring function
 */
typedef double (*ScoringFunc)(const Matrix *y_true, const Matrix *y_pred);

CrossValResults* cross_val_score_custom(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed,
    ScoringFunc scorer
);

/**
 * Print cross-validation results
 */
void cross_val_results_print(const CrossValResults *results);

/**
 * Free cross-validation results
 */
void cross_val_results_free(CrossValResults *results);

/* ============================================
 * Leave-One-Out Cross-Validation
 * ============================================ */

/**
 * Leave-One-Out cross-validation
 * Each sample is used once as test set
 */
CrossValResults* leave_one_out_cv(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y
);

/* ============================================
 * Train/Test Split utilities
 * ============================================ */

/**
 * Get subset of matrix by row indices
 */
Matrix* matrix_get_rows(const Matrix *m, const size_t *indices, size_t n_indices);

#endif /* VALIDATION_H */
