/**
 * validation.c - Cross-validation implementation
 */

#include "validation.h"
#include "utils.h"
#include "metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Helper functions
 * ============================================ */

Matrix* matrix_get_rows(const Matrix *m, const size_t *indices, size_t n_indices) {
    if (!m || !indices || n_indices == 0) return NULL;

    Matrix *result = matrix_alloc(n_indices, m->cols);
    if (!result) return NULL;

    for (size_t i = 0; i < n_indices; i++) {
        size_t src_row = indices[i];
        if (src_row >= m->rows) {
            matrix_free(result);
            return NULL;
        }
        for (size_t j = 0; j < m->cols; j++) {
            result->data[i * m->cols + j] = m->data[src_row * m->cols + j];
        }
    }

    return result;
}

static double array_mean(const double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum / n;
}

static double array_std(const double *arr, int n) {
    double m = array_mean(arr, n);
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = arr[i] - m;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

/* ============================================
 * K-Fold Cross-Validation
 * ============================================ */

KFold* kfold_create(int n_splits, int shuffle, unsigned int seed) {
    if (n_splits < 2) return NULL;

    KFold *kf = calloc(1, sizeof(KFold));
    if (!kf) return NULL;

    kf->n_splits = n_splits;
    kf->shuffle = shuffle;
    kf->seed = seed;
    kf->n_samples = 0;
    kf->indices = NULL;
    kf->current_fold = 0;

    return kf;
}

void kfold_init(KFold *kf, size_t n_samples) {
    if (!kf || n_samples == 0) return;

    kf->n_samples = n_samples;

    // Free old indices if any
    free(kf->indices);

    // Allocate indices
    kf->indices = malloc(n_samples * sizeof(size_t));
    if (!kf->indices) return;

    // Initialize indices
    for (size_t i = 0; i < n_samples; i++) {
        kf->indices[i] = i;
    }

    // Shuffle if requested
    if (kf->shuffle) {
        rand_seed(kf->seed);
        shuffle_indices(kf->indices, n_samples);
    }
}

FoldIndices kfold_get_fold(const KFold *kf, int fold) {
    FoldIndices fi = {NULL, 0, NULL, 0};

    if (!kf || !kf->indices || fold < 0 || fold >= kf->n_splits) {
        return fi;
    }

    size_t n = kf->n_samples;
    int k = kf->n_splits;

    // Calculate fold boundaries
    size_t fold_size = n / k;
    size_t remainder = n % k;

    size_t test_start = fold * fold_size + (fold < (int)remainder ? fold : remainder);
    size_t test_end = test_start + fold_size + (fold < (int)remainder ? 1 : 0);
    size_t n_test = test_end - test_start;
    size_t n_train = n - n_test;

    // Allocate arrays
    fi.test_indices = malloc(n_test * sizeof(size_t));
    fi.train_indices = malloc(n_train * sizeof(size_t));

    if (!fi.test_indices || !fi.train_indices) {
        free(fi.test_indices);
        free(fi.train_indices);
        fi.test_indices = NULL;
        fi.train_indices = NULL;
        return fi;
    }

    fi.n_test = n_test;
    fi.n_train = n_train;

    // Fill test indices
    for (size_t i = 0; i < n_test; i++) {
        fi.test_indices[i] = kf->indices[test_start + i];
    }

    // Fill train indices (everything except test)
    size_t train_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < test_start || i >= test_end) {
            fi.train_indices[train_idx++] = kf->indices[i];
        }
    }

    return fi;
}

void fold_indices_free(FoldIndices *fi) {
    if (fi) {
        free(fi->train_indices);
        free(fi->test_indices);
        fi->train_indices = NULL;
        fi->test_indices = NULL;
        fi->n_train = 0;
        fi->n_test = 0;
    }
}

void kfold_free(KFold *kf) {
    if (kf) {
        free(kf->indices);
        free(kf);
    }
}

/* ============================================
 * Stratified K-Fold
 * ============================================ */

StratifiedKFold* stratified_kfold_create(int n_splits, int shuffle, unsigned int seed) {
    if (n_splits < 2) return NULL;

    StratifiedKFold *skf = calloc(1, sizeof(StratifiedKFold));
    if (!skf) return NULL;

    skf->n_splits = n_splits;
    skf->shuffle = shuffle;
    skf->seed = seed;
    skf->n_samples = 0;
    skf->indices = NULL;
    skf->y = NULL;
    skf->n_classes = 0;

    return skf;
}

void stratified_kfold_init(StratifiedKFold *skf, const Matrix *y) {
    if (!skf || !y) return;

    skf->y = y;
    skf->n_samples = y->rows;

    // Find unique classes and count them
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    skf->n_classes = max_class + 1;

    // Count samples per class
    size_t *class_counts = calloc(skf->n_classes, sizeof(size_t));
    if (!class_counts) return;

    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        class_counts[label]++;
    }

    // Create indices array grouped by class
    size_t **class_indices = malloc(skf->n_classes * sizeof(size_t*));
    size_t *class_positions = calloc(skf->n_classes, sizeof(size_t));

    if (!class_indices || !class_positions) {
        free(class_counts);
        free(class_indices);
        free(class_positions);
        return;
    }

    for (int c = 0; c < skf->n_classes; c++) {
        class_indices[c] = malloc(class_counts[c] * sizeof(size_t));
        if (!class_indices[c]) {
            for (int j = 0; j < c; j++) free(class_indices[j]);
            free(class_indices);
            free(class_counts);
            free(class_positions);
            return;
        }
    }

    // Fill class indices
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        class_indices[label][class_positions[label]++] = i;
    }

    // Shuffle within each class if requested
    if (skf->shuffle) {
        rand_seed(skf->seed);
        for (int c = 0; c < skf->n_classes; c++) {
            shuffle_indices(class_indices[c], class_counts[c]);
        }
    }

    // Interleave indices from each class
    free(skf->indices);
    skf->indices = malloc(skf->n_samples * sizeof(size_t));
    if (!skf->indices) {
        for (int c = 0; c < skf->n_classes; c++) free(class_indices[c]);
        free(class_indices);
        free(class_counts);
        free(class_positions);
        return;
    }

    size_t idx = 0;
    memset(class_positions, 0, skf->n_classes * sizeof(size_t));

    // Round-robin from each class
    int done = 0;
    while (!done) {
        done = 1;
        for (int c = 0; c < skf->n_classes; c++) {
            if (class_positions[c] < class_counts[c]) {
                skf->indices[idx++] = class_indices[c][class_positions[c]++];
                done = 0;
            }
        }
    }

    // Cleanup
    for (int c = 0; c < skf->n_classes; c++) free(class_indices[c]);
    free(class_indices);
    free(class_counts);
    free(class_positions);
}

FoldIndices stratified_kfold_get_fold(const StratifiedKFold *skf, int fold) {
    // Similar to kfold but uses stratified indices
    FoldIndices fi = {NULL, 0, NULL, 0};

    if (!skf || !skf->indices || fold < 0 || fold >= skf->n_splits) {
        return fi;
    }

    size_t n = skf->n_samples;
    int k = skf->n_splits;

    size_t fold_size = n / k;
    size_t remainder = n % k;

    size_t test_start = fold * fold_size + (fold < (int)remainder ? fold : remainder);
    size_t test_end = test_start + fold_size + (fold < (int)remainder ? 1 : 0);
    size_t n_test = test_end - test_start;
    size_t n_train = n - n_test;

    fi.test_indices = malloc(n_test * sizeof(size_t));
    fi.train_indices = malloc(n_train * sizeof(size_t));

    if (!fi.test_indices || !fi.train_indices) {
        free(fi.test_indices);
        free(fi.train_indices);
        fi.test_indices = NULL;
        fi.train_indices = NULL;
        return fi;
    }

    fi.n_test = n_test;
    fi.n_train = n_train;

    for (size_t i = 0; i < n_test; i++) {
        fi.test_indices[i] = skf->indices[test_start + i];
    }

    size_t train_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < test_start || i >= test_end) {
            fi.train_indices[train_idx++] = skf->indices[i];
        }
    }

    return fi;
}

void stratified_kfold_free(StratifiedKFold *skf) {
    if (skf) {
        free(skf->indices);
        free(skf);
    }
}

/* ============================================
 * Cross-validation scoring
 * ============================================ */

CrossValResults* cross_val_score(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
) {
    if (!estimator || !X || !y || n_splits < 2) return NULL;

    CrossValResults *results = calloc(1, sizeof(CrossValResults));
    if (!results) return NULL;

    results->n_splits = n_splits;
    results->test_scores = malloc(n_splits * sizeof(double));
    results->train_scores = malloc(n_splits * sizeof(double));
    results->fit_times = malloc(n_splits * sizeof(double));
    results->score_times = malloc(n_splits * sizeof(double));

    if (!results->test_scores || !results->train_scores ||
        !results->fit_times || !results->score_times) {
        cross_val_results_free(results);
        return NULL;
    }

    // Create K-Fold
    KFold *kf = kfold_create(n_splits, shuffle, seed);
    if (!kf) {
        cross_val_results_free(results);
        return NULL;
    }
    kfold_init(kf, X->rows);

    // Iterate over folds
    for (int fold = 0; fold < n_splits; fold++) {
        FoldIndices fi = kfold_get_fold(kf, fold);
        if (!fi.train_indices || !fi.test_indices) {
            fold_indices_free(&fi);
            continue;
        }

        // Get train/test data for this fold
        Matrix *X_train = matrix_get_rows(X, fi.train_indices, fi.n_train);
        Matrix *y_train = matrix_get_rows(y, fi.train_indices, fi.n_train);
        Matrix *X_test = matrix_get_rows(X, fi.test_indices, fi.n_test);
        Matrix *y_test = matrix_get_rows(y, fi.test_indices, fi.n_test);

        if (!X_train || !y_train || !X_test || !y_test) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        // Clone estimator
        Estimator *clone = estimator->clone(estimator);
        if (!clone) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        // Fit
        clock_t fit_start = clock();
        clone->fit(clone, X_train, y_train);
        clock_t fit_end = clock();
        results->fit_times[fold] = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

        // Score
        clock_t score_start = clock();
        results->train_scores[fold] = clone->score(clone, X_train, y_train);
        results->test_scores[fold] = clone->score(clone, X_test, y_test);
        clock_t score_end = clock();
        results->score_times[fold] = (double)(score_end - score_start) / CLOCKS_PER_SEC;

        // Cleanup
        clone->free(clone);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        fold_indices_free(&fi);
    }

    kfold_free(kf);

    // Calculate summary statistics
    results->mean_test_score = array_mean(results->test_scores, n_splits);
    results->std_test_score = array_std(results->test_scores, n_splits);
    results->mean_train_score = array_mean(results->train_scores, n_splits);
    results->std_train_score = array_std(results->train_scores, n_splits);

    return results;
}

CrossValResults* cross_val_score_stratified(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
) {
    if (!estimator || !X || !y || n_splits < 2) return NULL;

    CrossValResults *results = calloc(1, sizeof(CrossValResults));
    if (!results) return NULL;

    results->n_splits = n_splits;
    results->test_scores = malloc(n_splits * sizeof(double));
    results->train_scores = malloc(n_splits * sizeof(double));
    results->fit_times = malloc(n_splits * sizeof(double));
    results->score_times = malloc(n_splits * sizeof(double));

    if (!results->test_scores || !results->train_scores ||
        !results->fit_times || !results->score_times) {
        cross_val_results_free(results);
        return NULL;
    }

    // Create Stratified K-Fold
    StratifiedKFold *skf = stratified_kfold_create(n_splits, shuffle, seed);
    if (!skf) {
        cross_val_results_free(results);
        return NULL;
    }
    stratified_kfold_init(skf, y);

    // Iterate over folds
    for (int fold = 0; fold < n_splits; fold++) {
        FoldIndices fi = stratified_kfold_get_fold(skf, fold);
        if (!fi.train_indices || !fi.test_indices) {
            fold_indices_free(&fi);
            continue;
        }

        Matrix *X_train = matrix_get_rows(X, fi.train_indices, fi.n_train);
        Matrix *y_train = matrix_get_rows(y, fi.train_indices, fi.n_train);
        Matrix *X_test = matrix_get_rows(X, fi.test_indices, fi.n_test);
        Matrix *y_test = matrix_get_rows(y, fi.test_indices, fi.n_test);

        if (!X_train || !y_train || !X_test || !y_test) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        Estimator *clone = estimator->clone(estimator);
        if (!clone) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        clock_t fit_start = clock();
        clone->fit(clone, X_train, y_train);
        clock_t fit_end = clock();
        results->fit_times[fold] = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

        clock_t score_start = clock();
        results->train_scores[fold] = clone->score(clone, X_train, y_train);
        results->test_scores[fold] = clone->score(clone, X_test, y_test);
        clock_t score_end = clock();
        results->score_times[fold] = (double)(score_end - score_start) / CLOCKS_PER_SEC;

        clone->free(clone);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        fold_indices_free(&fi);
    }

    stratified_kfold_free(skf);

    results->mean_test_score = array_mean(results->test_scores, n_splits);
    results->std_test_score = array_std(results->test_scores, n_splits);
    results->mean_train_score = array_mean(results->train_scores, n_splits);
    results->std_train_score = array_std(results->train_scores, n_splits);

    return results;
}

void cross_val_results_print(const CrossValResults *results) {
    if (!results) return;

    printf("\n=== Cross-Validation Results ===\n");
    printf("Number of folds: %d\n\n", results->n_splits);

    printf("%-8s %-15s %-15s %-12s\n", "Fold", "Train Score", "Test Score", "Fit Time");
    printf("----------------------------------------------------\n");

    for (int i = 0; i < results->n_splits; i++) {
        printf("%-8d %-15.6f %-15.6f %-12.4f\n",
               i + 1,
               results->train_scores[i],
               results->test_scores[i],
               results->fit_times[i]);
    }

    printf("----------------------------------------------------\n");
    printf("Mean test score:  %.6f (+/- %.6f)\n",
           results->mean_test_score, results->std_test_score * 2);
    printf("Mean train score: %.6f (+/- %.6f)\n",
           results->mean_train_score, results->std_train_score * 2);
    printf("================================\n\n");
}

void cross_val_results_free(CrossValResults *results) {
    if (results) {
        free(results->test_scores);
        free(results->train_scores);
        free(results->fit_times);
        free(results->score_times);
        free(results);
    }
}

/* ============================================
 * Leave-One-Out Cross-Validation
 * ============================================ */

CrossValResults* leave_one_out_cv(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y
) {
    return cross_val_score(estimator, X, y, (int)X->rows, 0, 0);
}
