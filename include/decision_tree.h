/**
 * decision_tree.h - Decision Tree for classification and regression
 *
 * Implements CART (Classification and Regression Trees):
 * - Binary splits using Gini impurity or MSE
 * - Recursive tree building
 * - Pruning support
 */

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "matrix.h"
#include "estimator.h"

/**
 * Criterion for splitting
 */
typedef enum {
    CRITERION_GINI,     // Gini impurity (classification)
    CRITERION_ENTROPY,  // Information gain (classification)
    CRITERION_MSE,      // Mean squared error (regression)
    CRITERION_MAE       // Mean absolute error (regression)
} SplitCriterion;

/**
 * Tree node structure
 */
typedef struct TreeNode {
    int is_leaf;

    // Split info (for internal nodes)
    size_t feature_index;
    double threshold;
    struct TreeNode *left;
    struct TreeNode *right;

    // Prediction info (for leaf nodes)
    double value;           // For regression: mean value
    int class_label;        // For classification: majority class
    double *class_proba;    // Class probabilities
    int n_classes;

    // Statistics
    size_t n_samples;
    double impurity;
} TreeNode;

/**
 * Decision Tree Classifier
 */
typedef struct {
    Estimator base;

    // Tree structure
    TreeNode *root;

    // Hyperparameters
    SplitCriterion criterion;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    double min_impurity_decrease;
    int max_features;       // 0 = all features

    // Training info
    int n_classes;
    int n_features;
    int depth_;
    int n_nodes_;
    int n_leaves_;
} DecisionTreeClassifier;

/**
 * Decision Tree Regressor
 */
typedef struct {
    Estimator base;

    TreeNode *root;

    SplitCriterion criterion;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    double min_impurity_decrease;
    int max_features;

    int n_features;
    int depth_;
    int n_nodes_;
    int n_leaves_;
} DecisionTreeRegressor;

/* ============================================
 * Decision Tree Classifier API
 * ============================================ */

/**
 * Create a Decision Tree Classifier
 */
DecisionTreeClassifier* decision_tree_classifier_create(void);

/**
 * Create with full configuration
 */
DecisionTreeClassifier* decision_tree_classifier_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
);

/**
 * Fit the classifier
 */
Estimator* decision_tree_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict class labels
 */
Matrix* decision_tree_classifier_predict(const Estimator *self, const Matrix *X);

/**
 * Predict class probabilities
 */
Matrix* decision_tree_classifier_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score (accuracy)
 */
double decision_tree_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* decision_tree_classifier_clone(const Estimator *self);

/**
 * Free
 */
void decision_tree_classifier_free(Estimator *self);

/**
 * Print tree structure
 */
void decision_tree_classifier_print_tree(const DecisionTreeClassifier *tree);

/**
 * Get feature importances
 */
Matrix* decision_tree_get_feature_importances(const DecisionTreeClassifier *tree);

/* ============================================
 * Decision Tree Regressor API
 * ============================================ */

/**
 * Create a Decision Tree Regressor
 */
DecisionTreeRegressor* decision_tree_regressor_create(void);

/**
 * Create with full configuration
 */
DecisionTreeRegressor* decision_tree_regressor_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
);

/**
 * Fit the regressor
 */
Estimator* decision_tree_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict values
 */
Matrix* decision_tree_regressor_predict(const Estimator *self, const Matrix *X);

/**
 * Score (R²)
 */
double decision_tree_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* decision_tree_regressor_clone(const Estimator *self);

/**
 * Free
 */
void decision_tree_regressor_free(Estimator *self);

/* ============================================
 * Tree utilities
 * ============================================ */

/**
 * Free a tree node recursively
 */
void tree_node_free(TreeNode *node);

/**
 * Export tree to text format
 */
void decision_tree_export_text(const TreeNode *root, int depth);

#endif /* DECISION_TREE_H */
