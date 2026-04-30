/**
 * decision_tree.c - Decision Tree implementation (CART algorithm)
 */

#include "decision_tree.h"
#include "cml_error.h"
#include "metrics.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ============================================
 * Helper functions
 * ============================================ */

static double gini_impurity(const double *y, const size_t *indices, size_t n, int n_classes) {
    if (n == 0) return 0.0;

    // Count each class
    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 1.0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    // Gini = 1 - sum(p_i^2)
    double gini = 1.0;
    for (int c = 0; c < n_classes; c++) {
        double p = (double)counts[c] / n;
        gini -= p * p;
    }

    free(counts);
    return gini;
}

static double entropy_impurity(const double *y, const size_t *indices, size_t n, int n_classes) {
    if (n == 0) return 0.0;

    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 0.0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    double ent = 0.0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] > 0) {
            double p = (double)counts[c] / n;
            ent -= p * log2(p);
        }
    }

    free(counts);
    return ent;
}

static double mse_impurity(const double *y, const size_t *indices, size_t n) {
    if (n == 0) return 0.0;

    // Calculate mean
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += y[indices[i]];
    }
    mean /= n;

    // Calculate MSE
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = y[indices[i]] - mean;
        mse += diff * diff;
    }

    return mse / n;
}

static int majority_class(const double *y, const size_t *indices, size_t n, int n_classes) {
    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    int max_count = 0;
    int max_class = 0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] > max_count) {
            max_count = counts[c];
            max_class = c;
        }
    }

    free(counts);
    return max_class;
}

static double mean_value(const double *y, const size_t *indices, size_t n) {
    if (n == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += y[indices[i]];
    }
    return sum / n;
}

static double* class_probabilities(const double *y, const size_t *indices, size_t n, int n_classes) {
    double *proba = calloc(n_classes, sizeof(double));
    if (!proba || n == 0) return proba;

    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) {
        free(proba);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    for (int c = 0; c < n_classes; c++) {
        proba[c] = (double)counts[c] / n;
    }

    free(counts);
    return proba;
}

/* ============================================
 * Tree building (Classification)
 * ============================================ */

typedef struct {
    size_t feature_index;
    double threshold;
    double impurity_decrease;
    size_t *left_indices;
    size_t n_left;
    size_t *right_indices;
    size_t n_right;
} BestSplit;

static BestSplit find_best_split_classification(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int n_classes,
    SplitCriterion criterion
) {
    BestSplit best = {0, 0.0, -DBL_MAX, NULL, 0, NULL, 0};

    double (*impurity_func)(const double*, const size_t*, size_t, int);
    if (criterion == CRITERION_ENTROPY) {
        impurity_func = entropy_impurity;
    } else {
        impurity_func = gini_impurity;
    }

    double parent_impurity = impurity_func(y, indices, n, n_classes);

    // Try each feature
    for (size_t feat = 0; feat < X->cols; feat++) {
        // Get unique values and sort
        double *values = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            values[i] = X->data[indices[i] * X->cols + feat];
        }

        // Sort values for threshold search
        qsort(values, n, sizeof(double), compare_doubles);

        // Pre-allocate split buffers outside the threshold loop
        size_t *left = malloc(n * sizeof(size_t));
        size_t *right = malloc(n * sizeof(size_t));
        if (!left || !right) {
            free(values);
            free(left);
            free(right);
            continue;
        }

        // Try midpoints as thresholds
        for (size_t t = 0; t < n - 1; t++) {
            if (values[t] == values[t + 1]) continue;  // Skip duplicates

            double threshold = (values[t] + values[t + 1]) / 2.0;

            // Split indices (reuse pre-allocated buffers)
            size_t n_left = 0, n_right = 0;

            for (size_t i = 0; i < n; i++) {
                double val = X->data[indices[i] * X->cols + feat];
                if (val <= threshold) {
                    left[n_left++] = indices[i];
                } else {
                    right[n_right++] = indices[i];
                }
            }

            if (n_left == 0 || n_right == 0) {
                continue;
            }

            // Calculate impurity decrease
            double left_impurity = impurity_func(y, left, n_left, n_classes);
            double right_impurity = impurity_func(y, right, n_right, n_classes);
            double weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n;
            double impurity_decrease = parent_impurity - weighted_impurity;

            if (impurity_decrease > best.impurity_decrease) {
                free(best.left_indices);
                free(best.right_indices);

                best.feature_index = feat;
                best.threshold = threshold;
                best.impurity_decrease = impurity_decrease;
                best.n_left = n_left;
                best.n_right = n_right;

                // Allocate new buffers for best, reuse left/right for next iteration
                best.left_indices = malloc(n_left * sizeof(size_t));
                best.right_indices = malloc(n_right * sizeof(size_t));
                if (best.left_indices) memcpy(best.left_indices, left, n_left * sizeof(size_t));
                if (best.right_indices) memcpy(best.right_indices, right, n_right * sizeof(size_t));
            }
        }

        free(left);
        free(right);
        free(values);
    }

    return best;
}

static TreeNode* build_tree_classification(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int n_classes,
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int depth,
    int *n_nodes
) {
    TreeNode *node = calloc(1, sizeof(TreeNode));
    if (!node) return NULL;

    (*n_nodes)++;
    node->n_samples = n;
    node->n_classes = n_classes;

    double (*impurity_func)(const double*, const size_t*, size_t, int);
    impurity_func = (criterion == CRITERION_ENTROPY) ? entropy_impurity : gini_impurity;
    node->impurity = impurity_func(y, indices, n, n_classes);

    // Check stopping criteria
    int should_stop = (
        depth >= max_depth ||
        (int)n < min_samples_split ||
        node->impurity < 1e-10  // Pure node
    );

    if (should_stop) {
        node->is_leaf = 1;
        node->class_label = majority_class(y, indices, n, n_classes);
        node->class_proba = class_probabilities(y, indices, n, n_classes);
        return node;
    }

    // Find best split
    BestSplit best = find_best_split_classification(X, y, indices, n, n_classes, criterion);

    // Check if split is valid
    if (best.impurity_decrease < min_impurity_decrease ||
        (int)best.n_left < min_samples_leaf ||
        (int)best.n_right < min_samples_leaf) {
        node->is_leaf = 1;
        node->class_label = majority_class(y, indices, n, n_classes);
        node->class_proba = class_probabilities(y, indices, n, n_classes);
        free(best.left_indices);
        free(best.right_indices);
        return node;
    }

    // Build children
    node->is_leaf = 0;
    node->feature_index = best.feature_index;
    node->threshold = best.threshold;

    node->left = build_tree_classification(
        X, y, best.left_indices, best.n_left, n_classes,
        criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    node->right = build_tree_classification(
        X, y, best.right_indices, best.n_right, n_classes,
        criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    free(best.left_indices);
    free(best.right_indices);

    return node;
}

/* ============================================
 * Tree building (Regression)
 * ============================================ */

static BestSplit find_best_split_regression(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n
) {
    BestSplit best = {0, 0.0, -DBL_MAX, NULL, 0, NULL, 0};

    double parent_mse = mse_impurity(y, indices, n);

    for (size_t feat = 0; feat < X->cols; feat++) {
        double *values = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            values[i] = X->data[indices[i] * X->cols + feat];
        }

        // Sort values for threshold search
        qsort(values, n, sizeof(double), compare_doubles);

        // Pre-allocate split buffers outside the threshold loop
        size_t *left = malloc(n * sizeof(size_t));
        size_t *right = malloc(n * sizeof(size_t));
        if (!left || !right) {
            free(values);
            free(left);
            free(right);
            continue;
        }

        for (size_t t = 0; t < n - 1; t++) {
            if (values[t] == values[t + 1]) continue;

            double threshold = (values[t] + values[t + 1]) / 2.0;

            // Split indices (reuse pre-allocated buffers)
            size_t n_left = 0, n_right = 0;

            for (size_t i = 0; i < n; i++) {
                double val = X->data[indices[i] * X->cols + feat];
                if (val <= threshold) {
                    left[n_left++] = indices[i];
                } else {
                    right[n_right++] = indices[i];
                }
            }

            if (n_left == 0 || n_right == 0) {
                continue;
            }

            double left_mse = mse_impurity(y, left, n_left);
            double right_mse = mse_impurity(y, right, n_right);
            double weighted_mse = (n_left * left_mse + n_right * right_mse) / n;
            double impurity_decrease = parent_mse - weighted_mse;

            if (impurity_decrease > best.impurity_decrease) {
                free(best.left_indices);
                free(best.right_indices);

                best.feature_index = feat;
                best.threshold = threshold;
                best.impurity_decrease = impurity_decrease;
                best.n_left = n_left;
                best.n_right = n_right;

                // Allocate new buffers for best, reuse left/right for next iteration
                best.left_indices = malloc(n_left * sizeof(size_t));
                best.right_indices = malloc(n_right * sizeof(size_t));
                if (best.left_indices) memcpy(best.left_indices, left, n_left * sizeof(size_t));
                if (best.right_indices) memcpy(best.right_indices, right, n_right * sizeof(size_t));
            }
        }

        free(left);
        free(right);
        free(values);
    }

    return best;
}

static TreeNode* build_tree_regression(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int depth,
    int *n_nodes
) {
    TreeNode *node = calloc(1, sizeof(TreeNode));
    if (!node) return NULL;

    (*n_nodes)++;
    node->n_samples = n;
    node->impurity = mse_impurity(y, indices, n);

    int should_stop = (
        depth >= max_depth ||
        (int)n < min_samples_split ||
        node->impurity < 1e-10
    );

    if (should_stop) {
        node->is_leaf = 1;
        node->value = mean_value(y, indices, n);
        return node;
    }

    BestSplit best = find_best_split_regression(X, y, indices, n);

    if (best.impurity_decrease < min_impurity_decrease ||
        (int)best.n_left < min_samples_leaf ||
        (int)best.n_right < min_samples_leaf) {
        node->is_leaf = 1;
        node->value = mean_value(y, indices, n);
        free(best.left_indices);
        free(best.right_indices);
        return node;
    }

    node->is_leaf = 0;
    node->feature_index = best.feature_index;
    node->threshold = best.threshold;

    node->left = build_tree_regression(
        X, y, best.left_indices, best.n_left,
        max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    node->right = build_tree_regression(
        X, y, best.right_indices, best.n_right,
        max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    free(best.left_indices);
    free(best.right_indices);

    return node;
}

/* ============================================
 * Decision Tree Classifier API
 * ============================================ */

DecisionTreeClassifier* decision_tree_classifier_create(void) {
    return decision_tree_classifier_create_full(CRITERION_GINI, 10, 2, 1, 0.0);
}

DecisionTreeClassifier* decision_tree_classifier_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
) {
    DecisionTreeClassifier *tree = calloc(1, sizeof(DecisionTreeClassifier));
    if (!tree) return NULL;

    tree->base.type = MODEL_DECISION_TREE;
    tree->base.task = TASK_CLASSIFICATION;
    tree->base.is_fitted = 0;
    tree->base.verbose = VERBOSE_SILENT;

    tree->base.fit = decision_tree_classifier_fit;
    tree->base.predict = decision_tree_classifier_predict;
    tree->base.predict_proba = decision_tree_classifier_predict_proba;
    tree->base.transform = NULL;
    tree->base.score = decision_tree_classifier_score;
    tree->base.clone = decision_tree_classifier_clone;
    tree->base.free = decision_tree_classifier_free;
    tree->base.save = NULL;
    tree->base.load = NULL;
    tree->base.print_summary = NULL;

    tree->root = NULL;
    tree->criterion = criterion;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->min_impurity_decrease = min_impurity_decrease;
    tree->max_features = 0;

    return tree;
}

Estimator* decision_tree_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    DecisionTreeClassifier *tree = (DecisionTreeClassifier*)self;

    // Free previous tree
    if (tree->root) {
        tree_node_free(tree->root);
        tree->root = NULL;
    }

    // Count classes
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    tree->n_classes = max_class + 1;
    tree->n_features = X->cols;

    // Create indices
    size_t *indices = malloc(X->rows * sizeof(size_t));
    for (size_t i = 0; i < X->rows; i++) {
        indices[i] = i;
    }

    // Build tree
    tree->n_nodes_ = 0;
    tree->root = build_tree_classification(
        X, y->data, indices, X->rows, tree->n_classes,
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease, 0, &tree->n_nodes_
    );

    free(indices);
    tree->base.is_fitted = 1;

    return self;
}

static int predict_single_class(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->class_label;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_single_class(node->left, x);
    } else {
        return predict_single_class(node->right, x);
    }
}

Matrix* decision_tree_classifier_predict(const Estimator *self, const Matrix *X) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_classifier_predict: model not fitted");
        return NULL;
    }

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] = predict_single_class(tree->root, &X->data[i * X->cols]);
    }

    return predictions;
}

static const double* predict_proba_single(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->class_proba;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_proba_single(node->left, x);
    } else {
        return predict_proba_single(node->right, x);
    }
}

Matrix* decision_tree_classifier_predict_proba(const Estimator *self, const Matrix *X) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_classifier_predict_proba: model not fitted");
        return NULL;
    }

    Matrix *proba = matrix_alloc(X->rows, tree->n_classes);
    if (!proba) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        const double *p = predict_proba_single(tree->root, &X->data[i * X->cols]);
        for (int c = 0; c < tree->n_classes; c++) {
            proba->data[i * tree->n_classes + c] = p ? p[c] : 0.0;
        }
    }

    return proba;
}

double decision_tree_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

Estimator* decision_tree_classifier_clone(const Estimator *self) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;
    return (Estimator*)decision_tree_classifier_create_full(
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease
    );
}

void tree_node_free(TreeNode *node) {
    if (!node) return;
    tree_node_free(node->left);
    tree_node_free(node->right);
    free(node->class_proba);
    free(node);
}

void decision_tree_classifier_free(Estimator *self) {
    DecisionTreeClassifier *tree = (DecisionTreeClassifier*)self;
    if (tree) {
        tree_node_free(tree->root);
        free(tree);
    }
}

void decision_tree_export_text(const TreeNode *root, int depth) {
    if (!root) return;

    for (int i = 0; i < depth; i++) printf("  ");

    if (root->is_leaf) {
        printf("return class %d (samples=%zu)\n", root->class_label, root->n_samples);
    } else {
        printf("if feature[%zu] <= %.4f:\n", root->feature_index, root->threshold);
        decision_tree_export_text(root->left, depth + 1);

        for (int i = 0; i < depth; i++) printf("  ");
        printf("else:\n");
        decision_tree_export_text(root->right, depth + 1);
    }
}

/* ============================================
 * Decision Tree Regressor API
 * ============================================ */

DecisionTreeRegressor* decision_tree_regressor_create(void) {
    return decision_tree_regressor_create_full(CRITERION_MSE, 10, 2, 1, 0.0);
}

DecisionTreeRegressor* decision_tree_regressor_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
) {
    DecisionTreeRegressor *tree = calloc(1, sizeof(DecisionTreeRegressor));
    if (!tree) return NULL;

    tree->base.type = MODEL_DECISION_TREE;
    tree->base.task = TASK_REGRESSION;
    tree->base.is_fitted = 0;
    tree->base.verbose = VERBOSE_SILENT;

    tree->base.fit = decision_tree_regressor_fit;
    tree->base.predict = decision_tree_regressor_predict;
    tree->base.predict_proba = NULL;
    tree->base.transform = NULL;
    tree->base.score = decision_tree_regressor_score;
    tree->base.clone = decision_tree_regressor_clone;
    tree->base.free = decision_tree_regressor_free;
    tree->base.save = NULL;
    tree->base.load = NULL;
    tree->base.print_summary = NULL;

    tree->root = NULL;
    tree->criterion = criterion;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->min_impurity_decrease = min_impurity_decrease;
    tree->max_features = 0;

    return tree;
}

Estimator* decision_tree_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    DecisionTreeRegressor *tree = (DecisionTreeRegressor*)self;

    if (tree->root) {
        tree_node_free(tree->root);
        tree->root = NULL;
    }

    tree->n_features = X->cols;

    size_t *indices = malloc(X->rows * sizeof(size_t));
    for (size_t i = 0; i < X->rows; i++) {
        indices[i] = i;
    }

    tree->n_nodes_ = 0;
    tree->root = build_tree_regression(
        X, y->data, indices, X->rows,
        tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease, 0, &tree->n_nodes_
    );

    free(indices);
    tree->base.is_fitted = 1;

    return self;
}

static double predict_single_value(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->value;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_single_value(node->left, x);
    } else {
        return predict_single_value(node->right, x);
    }
}

Matrix* decision_tree_regressor_predict(const Estimator *self, const Matrix *X) {
    const DecisionTreeRegressor *tree = (const DecisionTreeRegressor*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_regressor_predict: model not fitted");
        return NULL;
    }

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] = predict_single_value(tree->root, &X->data[i * X->cols]);
    }

    return predictions;
}

double decision_tree_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* decision_tree_regressor_clone(const Estimator *self) {
    const DecisionTreeRegressor *tree = (const DecisionTreeRegressor*)self;
    return (Estimator*)decision_tree_regressor_create_full(
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease
    );
}

void decision_tree_regressor_free(Estimator *self) {
    DecisionTreeRegressor *tree = (DecisionTreeRegressor*)self;
    if (tree) {
        tree_node_free(tree->root);
        free(tree);
    }
}
