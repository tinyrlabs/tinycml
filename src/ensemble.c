/**
 * ensemble.c - Random Forest implementation
 */

#include "ensemble.h"
#include "metrics.h"
#include "utils.h"
#include "validation.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Bootstrap sampling
 * ============================================ */

static void bootstrap_sample(size_t n_samples, size_t *indices, size_t *oob_mask, unsigned int seed) {
    rand_seed(seed);

    // Initialize OOB mask to 1 (out of bag)
    for (size_t i = 0; i < n_samples; i++) {
        oob_mask[i] = 1;
    }

    // Sample with replacement
    for (size_t i = 0; i < n_samples; i++) {
        indices[i] = (size_t)(rand_uniform() * n_samples);
        oob_mask[indices[i]] = 0;  // Mark as in bag
    }
}

/* ============================================
 * Random Forest Classifier
 * ============================================ */

RandomForestClassifier* random_forest_classifier_create(int n_estimators) {
    return random_forest_classifier_create_full(n_estimators, 10, 2, 1, 0, 1, 42);
}

RandomForestClassifier* random_forest_classifier_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
) {
    RandomForestClassifier *rf = calloc(1, sizeof(RandomForestClassifier));
    if (!rf) return NULL;

    rf->base.type = MODEL_RANDOM_FOREST;
    rf->base.task = TASK_CLASSIFICATION;
    rf->base.is_fitted = 0;
    rf->base.verbose = VERBOSE_SILENT;

    rf->base.fit = random_forest_classifier_fit;
    rf->base.predict = random_forest_classifier_predict;
    rf->base.predict_proba = random_forest_classifier_predict_proba;
    rf->base.transform = NULL;
    rf->base.score = random_forest_classifier_score;
    rf->base.clone = random_forest_classifier_clone;
    rf->base.free = random_forest_classifier_free;
    rf->base.save = NULL;
    rf->base.load = NULL;
    rf->base.print_summary = random_forest_classifier_print_summary;

    rf->n_estimators = n_estimators;
    rf->max_depth = max_depth;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf = min_samples_leaf;
    rf->max_features = max_features;
    rf->bootstrap = bootstrap;
    rf->seed = seed;

    rf->trees = NULL;
    rf->n_classes = 0;
    rf->oob_score_ = 0.0;

    return rf;
}

Estimator* random_forest_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    RandomForestClassifier *rf = (RandomForestClassifier*)self;
    clock_t start = clock();

    size_t n_samples = X->rows;
    rf->n_features = X->cols;

    // Determine n_classes
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    rf->n_classes = max_class + 1;

    // Determine max_features
    int max_feat = rf->max_features;
    if (max_feat <= 0) {
        max_feat = (int)sqrt((double)rf->n_features);
        if (max_feat < 1) max_feat = 1;
    }

    // Allocate trees
    rf->trees = malloc(rf->n_estimators * sizeof(DecisionTreeClassifier*));
    if (!rf->trees) return NULL;

    // OOB predictions for score calculation
    int *oob_counts = calloc(n_samples, sizeof(int));
    int *oob_predictions = calloc(n_samples * rf->n_classes, sizeof(int));

    if (rf->base.verbose >= VERBOSE_MINIMAL) {
        printf("Fitting %d trees...\n", rf->n_estimators);
    }

    for (int t = 0; t < rf->n_estimators; t++) {
        // Bootstrap sample
        size_t *indices = malloc(n_samples * sizeof(size_t));
        size_t *oob_mask = malloc(n_samples * sizeof(size_t));

        if (rf->bootstrap) {
            bootstrap_sample(n_samples, indices, oob_mask, rf->seed + t);
        } else {
            for (size_t i = 0; i < n_samples; i++) {
                indices[i] = i;
                oob_mask[i] = 0;
            }
        }

        // Create bootstrap sample
        Matrix *X_boot = matrix_get_rows(X, indices, n_samples);
        Matrix *y_boot = matrix_get_rows(y, indices, n_samples);

        // Create and fit tree
        rf->trees[t] = decision_tree_classifier_create_full(
            CRITERION_GINI,
            rf->max_depth,
            rf->min_samples_split,
            rf->min_samples_leaf,
            0.0
        );
        rf->trees[t]->max_features = max_feat;

        rf->trees[t]->base.fit((Estimator*)rf->trees[t], X_boot, y_boot);

        // OOB predictions
        if (rf->bootstrap) {
            for (size_t i = 0; i < n_samples; i++) {
                if (oob_mask[i]) {
                    Matrix *xi = matrix_alloc(1, X->cols);
                    for (size_t j = 0; j < X->cols; j++) {
                        xi->data[j] = X->data[i * X->cols + j];
                    }

                    Matrix *pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], xi);
                    if (pred) {
                        int pred_class = (int)pred->data[0];
                        oob_predictions[i * rf->n_classes + pred_class]++;
                        oob_counts[i]++;
                        matrix_free(pred);
                    }
                    matrix_free(xi);
                }
            }
        }

        matrix_free(X_boot);
        matrix_free(y_boot);
        free(indices);
        free(oob_mask);

        if (rf->base.verbose >= VERBOSE_PROGRESS && (t + 1) % 10 == 0) {
            printf("  [%d/%d] trees fitted\n", t + 1, rf->n_estimators);
        }
    }

    // Compute OOB score
    if (rf->bootstrap) {
        int correct = 0;
        int total = 0;
        for (size_t i = 0; i < n_samples; i++) {
            if (oob_counts[i] > 0) {
                int best_class = 0;
                int best_count = oob_predictions[i * rf->n_classes];
                for (int c = 1; c < rf->n_classes; c++) {
                    if (oob_predictions[i * rf->n_classes + c] > best_count) {
                        best_count = oob_predictions[i * rf->n_classes + c];
                        best_class = c;
                    }
                }
                if (best_class == (int)y->data[i]) {
                    correct++;
                }
                total++;
            }
        }
        rf->oob_score_ = total > 0 ? (double)correct / total : 0.0;
    }

    free(oob_counts);
    free(oob_predictions);

    rf->base.is_fitted = 1;

    if (rf->base.verbose >= VERBOSE_MINIMAL) {
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("RandomForest fitted in %.2f seconds\n", elapsed);
        if (rf->bootstrap) {
            printf("OOB score: %.4f\n", rf->oob_score_);
        }
    }

    return self;
}

Matrix* random_forest_classifier_predict(const Estimator *self, const Matrix *X) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    // Voting matrix
    int *votes = calloc(X->rows * rf->n_classes, sizeof(int));

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], X);
        if (tree_pred) {
            for (size_t i = 0; i < X->rows; i++) {
                int pred_class = (int)tree_pred->data[i];
                if (pred_class >= 0 && pred_class < rf->n_classes) {
                    votes[i * rf->n_classes + pred_class]++;
                }
            }
            matrix_free(tree_pred);
        }
    }

    // Majority vote
    for (size_t i = 0; i < X->rows; i++) {
        int best_class = 0;
        int best_count = votes[i * rf->n_classes];
        for (int c = 1; c < rf->n_classes; c++) {
            if (votes[i * rf->n_classes + c] > best_count) {
                best_count = votes[i * rf->n_classes + c];
                best_class = c;
            }
        }
        predictions->data[i] = best_class;
    }

    free(votes);
    return predictions;
}

Matrix* random_forest_classifier_predict_proba(const Estimator *self, const Matrix *X) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *proba = matrix_alloc(X->rows, rf->n_classes);
    if (!proba) return NULL;
    matrix_fill(proba, 0);

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_proba = rf->trees[t]->base.predict_proba((Estimator*)rf->trees[t], X);
        if (tree_proba) {
            for (size_t i = 0; i < proba->rows * proba->cols; i++) {
                proba->data[i] += tree_proba->data[i];
            }
            matrix_free(tree_proba);
        }
    }

    // Average
    for (size_t i = 0; i < proba->rows * proba->cols; i++) {
        proba->data[i] /= rf->n_estimators;
    }

    return proba;
}

double random_forest_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

Estimator* random_forest_classifier_clone(const Estimator *self) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;
    return (Estimator*)random_forest_classifier_create_full(
        rf->n_estimators, rf->max_depth, rf->min_samples_split,
        rf->min_samples_leaf, rf->max_features, rf->bootstrap, rf->seed
    );
}

void random_forest_classifier_free(Estimator *self) {
    RandomForestClassifier *rf = (RandomForestClassifier*)self;
    if (!rf) return;

    if (rf->trees) {
        for (int t = 0; t < rf->n_estimators; t++) {
            if (rf->trees[t]) {
                rf->trees[t]->base.free((Estimator*)rf->trees[t]);
            }
        }
        free(rf->trees);
    }
    free(rf);
}

void random_forest_classifier_print_summary(const Estimator *self) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    printf("\n=== RandomForestClassifier Summary ===\n");
    printf("Fitted: %s\n", rf->base.is_fitted ? "Yes" : "No");
    printf("Estimators: %d\n", rf->n_estimators);
    printf("Max depth: %d\n", rf->max_depth);
    printf("Max features: %d\n", rf->max_features);
    printf("Bootstrap: %s\n", rf->bootstrap ? "Yes" : "No");

    if (rf->base.is_fitted) {
        printf("Classes: %d\n", rf->n_classes);
        printf("Features: %d\n", rf->n_features);
        if (rf->bootstrap) {
            printf("OOB Score: %.4f\n", rf->oob_score_);
        }
    }
    printf("======================================\n\n");
}

/* ============================================
 * Random Forest Regressor
 * ============================================ */

RandomForestRegressor* random_forest_regressor_create(int n_estimators) {
    return random_forest_regressor_create_full(n_estimators, 10, 2, 1, 0, 1, 42);
}

RandomForestRegressor* random_forest_regressor_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
) {
    RandomForestRegressor *rf = calloc(1, sizeof(RandomForestRegressor));
    if (!rf) return NULL;

    rf->base.type = MODEL_RANDOM_FOREST;
    rf->base.task = TASK_REGRESSION;
    rf->base.is_fitted = 0;
    rf->base.verbose = VERBOSE_SILENT;

    rf->base.fit = random_forest_regressor_fit;
    rf->base.predict = random_forest_regressor_predict;
    rf->base.predict_proba = NULL;
    rf->base.transform = NULL;
    rf->base.score = random_forest_regressor_score;
    rf->base.clone = random_forest_regressor_clone;
    rf->base.free = random_forest_regressor_free;
    rf->base.save = NULL;
    rf->base.load = NULL;
    rf->base.print_summary = NULL;

    rf->n_estimators = n_estimators;
    rf->max_depth = max_depth;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf = min_samples_leaf;
    rf->max_features = max_features;
    rf->bootstrap = bootstrap;
    rf->seed = seed;

    rf->trees = NULL;
    rf->oob_score_ = 0.0;

    return rf;
}

Estimator* random_forest_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    RandomForestRegressor *rf = (RandomForestRegressor*)self;

    size_t n_samples = X->rows;
    rf->n_features = X->cols;

    int max_feat = rf->max_features;
    if (max_feat <= 0) {
        max_feat = rf->n_features;  // Use all features for regression
    }

    rf->trees = malloc(rf->n_estimators * sizeof(DecisionTreeRegressor*));
    if (!rf->trees) return NULL;

    for (int t = 0; t < rf->n_estimators; t++) {
        size_t *indices = malloc(n_samples * sizeof(size_t));
        size_t *oob_mask = malloc(n_samples * sizeof(size_t));

        if (rf->bootstrap) {
            bootstrap_sample(n_samples, indices, oob_mask, rf->seed + t);
        } else {
            for (size_t i = 0; i < n_samples; i++) {
                indices[i] = i;
            }
        }

        Matrix *X_boot = matrix_get_rows(X, indices, n_samples);
        Matrix *y_boot = matrix_get_rows(y, indices, n_samples);

        rf->trees[t] = decision_tree_regressor_create_full(
            CRITERION_MSE,
            rf->max_depth,
            rf->min_samples_split,
            rf->min_samples_leaf,
            0.0
        );
        rf->trees[t]->max_features = max_feat;

        rf->trees[t]->base.fit((Estimator*)rf->trees[t], X_boot, y_boot);

        matrix_free(X_boot);
        matrix_free(y_boot);
        free(indices);
        free(oob_mask);
    }

    rf->base.is_fitted = 1;
    return self;
}

Matrix* random_forest_regressor_predict(const Estimator *self, const Matrix *X) {
    const RandomForestRegressor *rf = (const RandomForestRegressor*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;
    matrix_fill(predictions, 0);

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], X);
        if (tree_pred) {
            for (size_t i = 0; i < X->rows; i++) {
                predictions->data[i] += tree_pred->data[i];
            }
            matrix_free(tree_pred);
        }
    }

    // Average
    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] /= rf->n_estimators;
    }

    return predictions;
}

double random_forest_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* random_forest_regressor_clone(const Estimator *self) {
    const RandomForestRegressor *rf = (const RandomForestRegressor*)self;
    return (Estimator*)random_forest_regressor_create_full(
        rf->n_estimators, rf->max_depth, rf->min_samples_split,
        rf->min_samples_leaf, rf->max_features, rf->bootstrap, rf->seed
    );
}

void random_forest_regressor_free(Estimator *self) {
    RandomForestRegressor *rf = (RandomForestRegressor*)self;
    if (!rf) return;

    if (rf->trees) {
        for (int t = 0; t < rf->n_estimators; t++) {
            if (rf->trees[t]) {
                rf->trees[t]->base.free((Estimator*)rf->trees[t]);
            }
        }
        free(rf->trees);
    }
    free(rf);
}
