/**
 * feature_selection.c - Feature Selection Utilities
 */

#include "feature_selection.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Statistical Helper Functions
 * ============================================ */

static double compute_mean(const double *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

static double compute_variance(const double *data, size_t n, double mean) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / n;
}

// Quickselect to find k-th smallest element index
static size_t partition_indices(double *scores, size_t *indices, size_t left, size_t right, size_t pivot_idx) {
    double pivot_val = scores[pivot_idx];

    // Swap pivot to end
    double tmp_s = scores[pivot_idx]; scores[pivot_idx] = scores[right]; scores[right] = tmp_s;
    size_t tmp_i = indices[pivot_idx]; indices[pivot_idx] = indices[right]; indices[right] = tmp_i;

    size_t store_idx = left;
    for (size_t i = left; i < right; i++) {
        if (scores[i] > pivot_val) {  // Descending order (higher scores first)
            tmp_s = scores[i]; scores[i] = scores[store_idx]; scores[store_idx] = tmp_s;
            tmp_i = indices[i]; indices[i] = indices[store_idx]; indices[store_idx] = tmp_i;
            store_idx++;
        }
    }

    // Swap pivot back
    tmp_s = scores[store_idx]; scores[store_idx] = scores[right]; scores[right] = tmp_s;
    tmp_i = indices[store_idx]; indices[store_idx] = indices[right]; indices[right] = tmp_i;

    return store_idx;
}

static void select_top_k_indices(double *scores, size_t *indices, size_t n, size_t k) {
    // Partial quicksort to get top k (descending)
    size_t left = 0, right = n - 1;

    while (left < right) {
        size_t pivot_idx = left + (right - left) / 2;
        pivot_idx = partition_indices(scores, indices, left, right, pivot_idx);

        if (pivot_idx == k - 1) {
            break;
        } else if (pivot_idx > k - 1) {
            right = pivot_idx - 1;
        } else {
            left = pivot_idx + 1;
        }
    }
}

/* ============================================
 * Scoring Functions
 * ============================================ */

int f_classif(const Matrix *X, const Matrix *y, double *f_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Count samples per class
    size_t *class_counts = calloc(n_classes, sizeof(size_t));
    for (size_t i = 0; i < n_samples; i++) {
        class_counts[(int)y->data[i]]++;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Compute overall mean
        double grand_mean = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            grand_mean += X->data[i * n_features + j];
        }
        grand_mean /= n_samples;

        // Compute class means
        double *class_means = calloc(n_classes, sizeof(double));
        for (size_t i = 0; i < n_samples; i++) {
            int label = (int)y->data[i];
            class_means[label] += X->data[i * n_features + j];
        }
        for (int c = 0; c < n_classes; c++) {
            if (class_counts[c] > 0) {
                class_means[c] /= class_counts[c];
            }
        }

        // Between-class variance (SSB)
        double ss_between = 0.0;
        for (int c = 0; c < n_classes; c++) {
            double diff = class_means[c] - grand_mean;
            ss_between += class_counts[c] * diff * diff;
        }

        // Within-class variance (SSW)
        double ss_within = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int label = (int)y->data[i];
            double diff = X->data[i * n_features + j] - class_means[label];
            ss_within += diff * diff;
        }

        free(class_means);

        // F-statistic = (SSB / df_between) / (SSW / df_within)
        int df_between = n_classes - 1;
        int df_within = n_samples - n_classes;

        if (df_within > 0 && ss_within > 0) {
            double ms_between = ss_between / df_between;
            double ms_within = ss_within / df_within;
            f_values[j] = ms_between / ms_within;
        } else {
            f_values[j] = 0.0;
        }

        // p-value approximation (simplified - would need F-distribution CDF)
        if (p_values) {
            p_values[j] = 0.0;  // Placeholder - full implementation needs F-dist CDF
        }
    }

    free(class_counts);
    return 0;
}

int f_regression(const Matrix *X, const Matrix *y, double *f_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Mean of y
    double y_mean = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        y_mean += y->data[i];
    }
    y_mean /= n_samples;

    // Variance of y
    double y_var = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        double diff = y->data[i] - y_mean;
        y_var += diff * diff;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Mean of feature
        double x_mean = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            x_mean += X->data[i * n_features + j];
        }
        x_mean /= n_samples;

        // Correlation coefficient
        double cov = 0.0;
        double x_var = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            double x_diff = X->data[i * n_features + j] - x_mean;
            double y_diff = y->data[i] - y_mean;
            cov += x_diff * y_diff;
            x_var += x_diff * x_diff;
        }

        double r = 0.0;
        if (x_var > 0 && y_var > 0) {
            r = cov / sqrt(x_var * y_var);
        }

        // F-statistic from correlation: F = r² * (n-2) / (1 - r²)
        double r2 = r * r;
        if (r2 < 1.0 && n_samples > 2) {
            f_values[j] = r2 * (n_samples - 2) / (1.0 - r2);
        } else {
            f_values[j] = 0.0;
        }

        if (p_values) {
            p_values[j] = 0.0;  // Placeholder
        }
    }

    return 0;
}

int chi2(const Matrix *X, const Matrix *y, double *chi2_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Sum per class
    double *class_sums = calloc(n_classes, sizeof(double));
    double total_sum = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++) {
            double val = X->data[i * n_features + j];
            if (val < 0) {
                free(class_sums);
                return -1;  // Chi2 requires non-negative features
            }
            class_sums[(int)y->data[i]] += val;
            total_sum += val;
        }
    }

    for (size_t j = 0; j < n_features; j++) {
        // Feature sum
        double feature_sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            feature_sum += X->data[i * n_features + j];
        }

        // Chi-squared for this feature
        double chi2_val = 0.0;
        for (int c = 0; c < n_classes; c++) {
            // Observed
            double observed = 0.0;
            for (size_t i = 0; i < n_samples; i++) {
                if ((int)y->data[i] == c) {
                    observed += X->data[i * n_features + j];
                }
            }

            // Expected = (feature_sum * class_sum) / total
            double expected = 0.0;
            if (total_sum > 0) {
                expected = (feature_sum * class_sums[c]) / total_sum;
            }

            if (expected > 0) {
                chi2_val += (observed - expected) * (observed - expected) / expected;
            }
        }

        chi2_values[j] = chi2_val;
        if (p_values) p_values[j] = 0.0;
    }

    free(class_sums);
    return 0;
}

int mutual_info_classif(const Matrix *X, const Matrix *y, double *mi_values, int n_bins) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    if (n_bins <= 0) n_bins = 10;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Class probabilities
    double *p_y = calloc(n_classes, sizeof(double));
    for (size_t i = 0; i < n_samples; i++) {
        p_y[(int)y->data[i]] += 1.0 / n_samples;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Find feature range
        double x_min = X->data[j];
        double x_max = X->data[j];
        for (size_t i = 1; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            if (val < x_min) x_min = val;
            if (val > x_max) x_max = val;
        }

        double bin_width = (x_max - x_min) / n_bins;
        if (bin_width < 1e-10) bin_width = 1.0;

        // Bin counts: p(x), p(x,y)
        double *p_x = calloc(n_bins, sizeof(double));
        double *p_xy = calloc(n_bins * n_classes, sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            int bin = (int)((val - x_min) / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;
            if (bin < 0) bin = 0;

            int label = (int)y->data[i];
            p_x[bin] += 1.0 / n_samples;
            p_xy[bin * n_classes + label] += 1.0 / n_samples;
        }

        // MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        double mi = 0.0;
        for (int b = 0; b < n_bins; b++) {
            for (int c = 0; c < n_classes; c++) {
                double pxy = p_xy[b * n_classes + c];
                double px = p_x[b];
                double py = p_y[c];

                if (pxy > 0 && px > 0 && py > 0) {
                    mi += pxy * log(pxy / (px * py));
                }
            }
        }

        mi_values[j] = mi > 0 ? mi : 0.0;

        free(p_x);
        free(p_xy);
    }

    free(p_y);
    return 0;
}

int mutual_info_regression(const Matrix *X, const Matrix *y, double *mi_values, int n_bins) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    if (n_bins <= 0) n_bins = 10;

    // Find y range
    double y_min = y->data[0];
    double y_max = y->data[0];
    for (size_t i = 1; i < n_samples; i++) {
        if (y->data[i] < y_min) y_min = y->data[i];
        if (y->data[i] > y_max) y_max = y->data[i];
    }
    double y_bin_width = (y_max - y_min) / n_bins;
    if (y_bin_width < 1e-10) y_bin_width = 1.0;

    // Bin y values
    int *y_bins = malloc(n_samples * sizeof(int));
    double *p_y = calloc(n_bins, sizeof(double));
    for (size_t i = 0; i < n_samples; i++) {
        int bin = (int)((y->data[i] - y_min) / y_bin_width);
        if (bin >= n_bins) bin = n_bins - 1;
        if (bin < 0) bin = 0;
        y_bins[i] = bin;
        p_y[bin] += 1.0 / n_samples;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Find feature range
        double x_min = X->data[j];
        double x_max = X->data[j];
        for (size_t i = 1; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            if (val < x_min) x_min = val;
            if (val > x_max) x_max = val;
        }

        double bin_width = (x_max - x_min) / n_bins;
        if (bin_width < 1e-10) bin_width = 1.0;

        double *p_x = calloc(n_bins, sizeof(double));
        double *p_xy = calloc(n_bins * n_bins, sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            int x_bin = (int)((val - x_min) / bin_width);
            if (x_bin >= n_bins) x_bin = n_bins - 1;
            if (x_bin < 0) x_bin = 0;

            p_x[x_bin] += 1.0 / n_samples;
            p_xy[x_bin * n_bins + y_bins[i]] += 1.0 / n_samples;
        }

        // MI calculation
        double mi = 0.0;
        for (int bx = 0; bx < n_bins; bx++) {
            for (int by = 0; by < n_bins; by++) {
                double pxy = p_xy[bx * n_bins + by];
                double px = p_x[bx];
                double py = p_y[by];

                if (pxy > 0 && px > 0 && py > 0) {
                    mi += pxy * log(pxy / (px * py));
                }
            }
        }

        mi_values[j] = mi > 0 ? mi : 0.0;

        free(p_x);
        free(p_xy);
    }

    free(y_bins);
    free(p_y);
    return 0;
}

/* ============================================
 * SelectKBest Implementation
 * ============================================ */

SelectKBest* select_k_best_create(ScoreFunction score_func, int k) {
    SelectKBest *skb = calloc(1, sizeof(SelectKBest));
    if (!skb) return NULL;

    skb->base.type = MODEL_FEATURE_SELECTOR;
    skb->base.task = TASK_TRANSFORMATION;
    skb->base.is_fitted = 0;
    skb->base.verbose = VERBOSE_SILENT;

    skb->base.fit = select_k_best_fit;
    skb->base.predict = NULL;
    skb->base.predict_proba = NULL;
    skb->base.transform = select_k_best_transform;
    skb->base.score = NULL;
    skb->base.clone = select_k_best_clone;
    skb->base.free = select_k_best_free;
    skb->base.save = NULL;
    skb->base.load = NULL;
    skb->base.print_summary = select_k_best_print_summary;

    skb->score_func = score_func;
    skb->k = k;
    skb->scores_ = NULL;
    skb->pvalues_ = NULL;
    skb->support_ = NULL;

    return skb;
}

Estimator* select_k_best_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    SelectKBest *skb = (SelectKBest*)self;

    skb->n_features_ = X->cols;
    int k = skb->k;
    if (k <= 0 || k > (int)X->cols) {
        k = X->cols;
    }
    skb->n_features_selected_ = k;

    // Allocate
    skb->scores_ = malloc(X->cols * sizeof(double));
    skb->pvalues_ = malloc(X->cols * sizeof(double));
    skb->support_ = calloc(X->cols, sizeof(int));

    // Compute scores
    switch (skb->score_func) {
        case SCORE_F_CLASSIF:
            f_classif(X, y, skb->scores_, skb->pvalues_);
            break;
        case SCORE_F_REGRESSION:
            f_regression(X, y, skb->scores_, skb->pvalues_);
            break;
        case SCORE_MUTUAL_INFO_CLASSIF:
            mutual_info_classif(X, y, skb->scores_, 10);
            break;
        case SCORE_MUTUAL_INFO_REGRESSION:
            mutual_info_regression(X, y, skb->scores_, 10);
            break;
        case SCORE_CHI2:
            chi2(X, y, skb->scores_, skb->pvalues_);
            break;
    }

    // Select top k features
    double *scores_copy = malloc(X->cols * sizeof(double));
    size_t *indices = malloc(X->cols * sizeof(size_t));
    for (size_t j = 0; j < X->cols; j++) {
        scores_copy[j] = skb->scores_[j];
        indices[j] = j;
    }

    select_top_k_indices(scores_copy, indices, X->cols, k);

    // Mark selected features
    for (int i = 0; i < k; i++) {
        skb->support_[indices[i]] = 1;
    }

    free(scores_copy);
    free(indices);

    skb->base.is_fitted = 1;

    if (skb->base.verbose >= VERBOSE_MINIMAL) {
        printf("SelectKBest: selected %d/%d features\n", k, skb->n_features_);
    }

    return self;
}

Matrix* select_k_best_transform(const Estimator *self, const Matrix *X) {
    const SelectKBest *skb = (const SelectKBest*)self;

    if (!skb->base.is_fitted) return NULL;

    Matrix *X_new = matrix_alloc(X->rows, skb->n_features_selected_);
    if (!X_new) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        size_t new_j = 0;
        for (int j = 0; j < skb->n_features_; j++) {
            if (skb->support_[j]) {
                X_new->data[i * skb->n_features_selected_ + new_j] =
                    X->data[i * X->cols + j];
                new_j++;
            }
        }
    }

    return X_new;
}

const double* select_k_best_scores(const SelectKBest *skb) {
    return skb ? skb->scores_ : NULL;
}

const int* select_k_best_get_support(const SelectKBest *skb) {
    return skb ? skb->support_ : NULL;
}

Estimator* select_k_best_clone(const Estimator *self) {
    const SelectKBest *skb = (const SelectKBest*)self;
    return (Estimator*)select_k_best_create(skb->score_func, skb->k);
}

void select_k_best_free(Estimator *self) {
    SelectKBest *skb = (SelectKBest*)self;
    if (!skb) return;

    free(skb->scores_);
    free(skb->pvalues_);
    free(skb->support_);
    free(skb);
}

void select_k_best_print_summary(const Estimator *self) {
    const SelectKBest *skb = (const SelectKBest*)self;

    const char *score_names[] = {
        "f_classif", "f_regression", "mutual_info_classif",
        "mutual_info_regression", "chi2"
    };

    printf("\n=== SelectKBest Summary ===\n");
    printf("Fitted: %s\n", skb->base.is_fitted ? "Yes" : "No");
    printf("Score function: %s\n", score_names[skb->score_func]);
    printf("k: %d\n", skb->k);

    if (skb->base.is_fitted) {
        printf("Features selected: %d/%d\n", skb->n_features_selected_, skb->n_features_);
        printf("\nFeature scores:\n");
        for (int j = 0; j < skb->n_features_ && j < 10; j++) {
            printf("  [%d] %.4f %s\n", j, skb->scores_[j],
                   skb->support_[j] ? "(selected)" : "");
        }
        if (skb->n_features_ > 10) {
            printf("  ... (%d more)\n", skb->n_features_ - 10);
        }
    }
    printf("===========================\n\n");
}

/* ============================================
 * VarianceThreshold Implementation
 * ============================================ */

VarianceThreshold* variance_threshold_create(double threshold) {
    VarianceThreshold *vt = calloc(1, sizeof(VarianceThreshold));
    if (!vt) return NULL;

    vt->base.type = MODEL_FEATURE_SELECTOR;
    vt->base.task = TASK_TRANSFORMATION;
    vt->base.is_fitted = 0;
    vt->base.verbose = VERBOSE_SILENT;

    vt->base.fit = variance_threshold_fit;
    vt->base.predict = NULL;
    vt->base.predict_proba = NULL;
    vt->base.transform = variance_threshold_transform;
    vt->base.score = NULL;
    vt->base.clone = variance_threshold_clone;
    vt->base.free = variance_threshold_free;
    vt->base.save = NULL;
    vt->base.load = NULL;
    vt->base.print_summary = NULL;

    vt->threshold = threshold;
    vt->variances_ = NULL;
    vt->support_ = NULL;

    return vt;
}

Estimator* variance_threshold_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;
    VarianceThreshold *vt = (VarianceThreshold*)self;

    vt->n_features_ = X->cols;
    vt->variances_ = malloc(X->cols * sizeof(double));
    vt->support_ = calloc(X->cols, sizeof(int));
    vt->n_features_selected_ = 0;

    for (size_t j = 0; j < X->cols; j++) {
        // Compute mean
        double mean = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            mean += X->data[i * X->cols + j];
        }
        mean /= X->rows;

        // Compute variance
        double var = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            double diff = X->data[i * X->cols + j] - mean;
            var += diff * diff;
        }
        var /= X->rows;

        vt->variances_[j] = var;
        if (var > vt->threshold) {
            vt->support_[j] = 1;
            vt->n_features_selected_++;
        }
    }

    vt->base.is_fitted = 1;

    if (vt->base.verbose >= VERBOSE_MINIMAL) {
        printf("VarianceThreshold: kept %d/%d features (threshold=%.4f)\n",
               vt->n_features_selected_, vt->n_features_, vt->threshold);
    }

    return self;
}

Matrix* variance_threshold_transform(const Estimator *self, const Matrix *X) {
    const VarianceThreshold *vt = (const VarianceThreshold*)self;

    if (!vt->base.is_fitted) return NULL;
    if (vt->n_features_selected_ == 0) return NULL;

    Matrix *X_new = matrix_alloc(X->rows, vt->n_features_selected_);
    if (!X_new) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        size_t new_j = 0;
        for (int j = 0; j < vt->n_features_; j++) {
            if (vt->support_[j]) {
                X_new->data[i * vt->n_features_selected_ + new_j] =
                    X->data[i * X->cols + j];
                new_j++;
            }
        }
    }

    return X_new;
}

const double* variance_threshold_variances(const VarianceThreshold *vt) {
    return vt ? vt->variances_ : NULL;
}

const int* variance_threshold_get_support(const VarianceThreshold *vt) {
    return vt ? vt->support_ : NULL;
}

Estimator* variance_threshold_clone(const Estimator *self) {
    const VarianceThreshold *vt = (const VarianceThreshold*)self;
    return (Estimator*)variance_threshold_create(vt->threshold);
}

void variance_threshold_free(Estimator *self) {
    VarianceThreshold *vt = (VarianceThreshold*)self;
    if (!vt) return;

    free(vt->variances_);
    free(vt->support_);
    free(vt);
}

/* ============================================
 * Utility Functions
 * ============================================ */

int get_feature_importances(const Estimator *estimator, double *importances) {
    if (!estimator || !estimator->is_fitted) return -1;

    switch (estimator->type) {
        case MODEL_LINEAR_REGRESSION: {
            // Use absolute coefficient values
            // Need to access internal weights - simplified version
            return -1;  // Would need access to internal structure
        }
        case MODEL_DECISION_TREE:
        case MODEL_RANDOM_FOREST:
            // Would need to compute from tree structure
            return -1;
        default:
            return -1;
    }
}
