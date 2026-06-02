/**
 * gradient_boosting.c - Gradient Boosted Decision Trees (GBDT)
 *
 * Implements gradient boosting for regression and binary classification
 * using DecisionTree as weak learner.
 */

#include "gradient_boosting.h"
#include "cml_serialization.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── helpers ─────────────────────────────────────────────────────── */

static double gb_sigmoid(double x) {
    if (x > 500.0) return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

static unsigned int gb_next_seed(unsigned int *s) {
    /* xorshift32 */
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* ── create / free ───────────────────────────────────────────────── */

GradientBoosting* gradient_boosting_create(
    int n_estimators,
    double learning_rate,
    int max_depth,
    int min_samples_split,
    double subsample)
{
    GradientBoosting *gb = calloc(1, sizeof(GradientBoosting));
    if (!gb) return NULL;

    gb->n_estimators      = n_estimators > 0 ? n_estimators : 100;
    gb->learning_rate     = learning_rate > 0 ? learning_rate : 0.1;
    gb->max_depth         = max_depth > 0 ? max_depth : 3;
    gb->min_samples_split = min_samples_split > 1 ? min_samples_split : 2;
    gb->subsample         = subsample > 0 && subsample <= 1.0 ? subsample : 1.0;
    gb->seed              = (unsigned int)time(NULL);

    gb->trees          = NULL;
    gb->init_prediction = 0.0;
    gb->n_classes       = 0;
    gb->n_features      = 0;

    /* Estimator vtable */
    gb->base.fit          = gradient_boosting_fit;
    gb->base.predict      = gradient_boosting_predict;
    gb->base.predict_proba = gradient_boosting_predict_proba;
    gb->base.score        = gradient_boosting_score;
    gb->base.clone        = gradient_boosting_clone;
    gb->base.free         = gradient_boosting_free;
    gb->base.type         = MODEL_GRADIENT_BOOSTING;

    return gb;
}

void gradient_boosting_free(Estimator *self) {
    if (!self) return;
    GradientBoosting *gb = (GradientBoosting*)self;
    if (gb->trees) {
        for (int t = 0; t < gb->n_estimators; t++) {
            if (gb->trees[t]) {
                gb->trees[t]->base.free((Estimator*)gb->trees[t]);
            }
        }
        free(gb->trees);
    }
    free(gb);
}

/* ── fit ─────────────────────────────────────────────────────────── */

Estimator* gradient_boosting_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y || X->rows != y->rows) return NULL;

    GradientBoosting *gb = (GradientBoosting*)self;
    size_t n = X->rows;
    gb->n_features = (int)X->cols;

    /* Determine number of classes from labels */
    int max_class = 0;
    for (size_t i = 0; i < n; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    gb->n_classes = max_class + 1;

    /* For simplicity and robustness: handle binary classification + regression */
    int is_classification = (gb->n_classes <= 2);

    /* Allocate trees */
    gb->trees = calloc(gb->n_estimators, sizeof(DecisionTreeRegressor*));
    if (!gb->trees) return NULL;

    /* Initial prediction */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += y->data[i];
    double mean_y = sum / (double)n;

    if (is_classification) {
        /* Log-odds of mean */
        double p = mean_y;
        if (p < 1e-10) p = 1e-10;
        if (p > 1.0 - 1e-10) p = 1.0 - 1e-10;
        gb->init_prediction = log(p / (1.0 - p));
    } else {
        gb->init_prediction = mean_y;
    }

    /* Current predictions */
    double *pred = malloc(n * sizeof(double));
    if (!pred) return NULL;
    for (size_t i = 0; i < n; i++) {
        pred[i] = gb->init_prediction;
    }

    /* Boosting rounds */
    for (int t = 0; t < gb->n_estimators; t++) {
        /* Compute pseudo-residuals */
        Matrix *residuals = matrix_alloc(n, 1);
        if (!residuals) { free(pred); return NULL; }

        if (is_classification) {
            for (size_t i = 0; i < n; i++) {
                double prob = gb_sigmoid(pred[i]);
                residuals->data[i] = y->data[i] - prob;
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                residuals->data[i] = y->data[i] - pred[i];
            }
        }

        /* Subsample if needed */
        Matrix *X_train = (Matrix*)X;
        Matrix *y_train = residuals;
        size_t *indices = NULL;
        size_t train_n = n;

        if (gb->subsample < 1.0) {
            train_n = (size_t)(n * gb->subsample);
            if (train_n < 2) train_n = 2;
            indices = malloc(train_n * sizeof(size_t));
            if (indices) {
                for (size_t i = 0; i < train_n; i++) {
                    indices[i] = gb_next_seed(&gb->seed) % n;
                }
                X_train = matrix_alloc(train_n, X->cols);
                y_train = matrix_alloc(train_n, 1);
                for (size_t i = 0; i < train_n; i++) {
                    size_t idx = indices[i];
                    for (size_t j = 0; j < X->cols; j++)
                        matrix_set(X_train, i, j, matrix_get(X, idx, j));
                    matrix_set(y_train, i, 0, matrix_get(residuals, idx, 0));
                }
            }
        }

        /* Fit weak learner (DecisionTreeRegressor, depth-limited) */
        gb->trees[t] = decision_tree_regressor_create_full(
            0,  /* CRITERION_MSE */
            gb->max_depth,
            gb->min_samples_split,
            1,  /* min_samples_leaf */
            0.0 /* min_impurity_decrease */
        );
        if (!gb->trees[t]) {
            matrix_free(residuals);
            free(pred);
            if (indices) free(indices);
            return NULL;
        }
        gb->trees[t]->base.fit((Estimator*)gb->trees[t], X_train, y_train);

        /* Update predictions */
        Matrix *tree_pred = gb->trees[t]->base.predict((Estimator*)gb->trees[t], X);
        if (tree_pred) {
            for (size_t i = 0; i < n; i++) {
                pred[i] += gb->learning_rate * tree_pred->data[i];
            }
            matrix_free(tree_pred);
        }

        /* Cleanup */
        if (gb->subsample < 1.0 && indices) {
            if (X_train != X) matrix_free(X_train);
            if (y_train != residuals) matrix_free(y_train);
            free(indices);
        }
        matrix_free(residuals);
    }

    free(pred);
    return self;
}

/* ── predict ─────────────────────────────────────────────────────── */

Matrix* gradient_boosting_predict(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    GradientBoosting *gb = (GradientBoosting*)self;

    size_t n = X->rows;
    int is_classification = (gb->n_classes <= 2);

    /* Compute raw predictions */
    double *raw = malloc(n * sizeof(double));
    if (!raw) return NULL;

    for (size_t i = 0; i < n; i++) raw[i] = gb->init_prediction;

    for (int t = 0; t < gb->n_estimators; t++) {
        if (!gb->trees[t]) continue;
        Matrix *tp = gb->trees[t]->base.predict((Estimator*)gb->trees[t], X);
        if (tp) {
            for (size_t i = 0; i < n; i++) {
                raw[i] += gb->learning_rate * tp->data[i];
            }
            matrix_free(tp);
        }
    }

    /* Convert to labels */
    Matrix *out = matrix_alloc(n, 1);
    if (!out) { free(raw); return NULL; }

    if (is_classification) {
        for (size_t i = 0; i < n; i++) {
            out->data[i] = gb_sigmoid(raw[i]) >= 0.5 ? 1.0 : 0.0;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            out->data[i] = raw[i];
        }
    }

    free(raw);
    return out;
}

/* ── predict_proba ───────────────────────────────────────────────── */

Matrix* gradient_boosting_predict_proba(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    GradientBoosting *gb = (GradientBoosting*)self;

    size_t n = X->rows;
    int is_classification = (gb->n_classes <= 2);

    if (!is_classification) {
        /* For regression, return predictions as probabilities */
        return gradient_boosting_predict(self, X);
    }

    /* Binary classification: return [prob_class0, prob_class1] */
    Matrix *out = matrix_alloc(n, 2);
    if (!out) return NULL;

    double *raw = malloc(n * sizeof(double));
    if (!raw) { matrix_free(out); return NULL; }

    for (size_t i = 0; i < n; i++) raw[i] = gb->init_prediction;

    for (int t = 0; t < gb->n_estimators; t++) {
        if (!gb->trees[t]) continue;
        Matrix *tp = gb->trees[t]->base.predict((Estimator*)gb->trees[t], X);
        if (tp) {
            for (size_t i = 0; i < n; i++) {
                raw[i] += gb->learning_rate * tp->data[i];
            }
            matrix_free(tp);
        }
    }

    for (size_t i = 0; i < n; i++) {
        double p1 = gb_sigmoid(raw[i]);
        matrix_set(out, i, 0, 1.0 - p1);
        matrix_set(out, i, 1, p1);
    }

    free(raw);
    return out;
}

/* ── score ───────────────────────────────────────────────────────── */

double gradient_boosting_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return -1.0;
    GradientBoosting *gb = (GradientBoosting*)self;

    Matrix *pred = gradient_boosting_predict(self, X);
    if (!pred) return -1.0;

    int is_classification = (gb->n_classes <= 2);

    if (is_classification) {
        /* Accuracy */
        size_t correct = 0;
        for (size_t i = 0; i < y->rows; i++) {
            int pred_label = (int)(pred->data[i] + 0.5);
            int true_label = (int)(y->data[i] + 0.5);
            if (pred_label == true_label) correct++;
        }
        matrix_free(pred);
        return (double)correct / (double)y->rows;
    } else {
        /* R² score */
        double mean_y = 0.0;
        for (size_t i = 0; i < y->rows; i++) mean_y += y->data[i];
        mean_y /= (double)y->rows;

        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i = 0; i < y->rows; i++) {
            double diff = y->data[i] - pred->data[i];
            ss_res += diff * diff;
            double diff_mean = y->data[i] - mean_y;
            ss_tot += diff_mean * diff_mean;
        }
        matrix_free(pred);
        if (ss_tot == 0.0) return 0.0;
        return 1.0 - ss_res / ss_tot;
    }
}

/* ── clone ───────────────────────────────────────────────────────── */

Estimator* gradient_boosting_clone(const Estimator *self) {
    if (!self) return NULL;
    GradientBoosting *src = (GradientBoosting*)self;

    GradientBoosting *dst = gradient_boosting_create(
        src->n_estimators,
        src->learning_rate,
        src->max_depth,
        src->min_samples_split,
        src->subsample
    );
    if (!dst) return NULL;

    dst->init_prediction = src->init_prediction;
    dst->n_classes = src->n_classes;
    dst->n_features = src->n_features;
    dst->seed = src->seed;

    if (src->trees) {
        dst->trees = calloc(src->n_estimators, sizeof(DecisionTreeRegressor*));
        if (!dst->trees) { free(dst); return NULL; }
        for (int t = 0; t < src->n_estimators; t++) {
            if (src->trees[t]) {
                Estimator *cloned = src->trees[t]->base.clone((Estimator*)src->trees[t]);
                dst->trees[t] = (DecisionTreeRegressor*)cloned;
            }
        }
    }

    return (Estimator*)dst;
}

/* ── save / load ─────────────────────────────────────────────────── */

#define GB_MAGIC "CML\0"
#define GB_SUBTYPE 10

int gradient_boosting_save(const Estimator *self, const char *path) {
    if (!self || !path) return -1;
    GradientBoosting *gb = (GradientBoosting*)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Write header */
    if (cml_ser_write_header(f, GB_SUBTYPE) != 0) {
        fclose(f);
        return -1;
    }

    /* Write hyperparameters */
    fwrite(&gb->n_estimators, sizeof(int), 1, f);
    fwrite(&gb->learning_rate, sizeof(double), 1, f);
    fwrite(&gb->max_depth, sizeof(int), 1, f);
    fwrite(&gb->min_samples_split, sizeof(int), 1, f);
    fwrite(&gb->subsample, sizeof(double), 1, f);
    fwrite(&gb->init_prediction, sizeof(double), 1, f);
    fwrite(&gb->n_classes, sizeof(int), 1, f);
    fwrite(&gb->n_features, sizeof(int), 1, f);
    fwrite(&gb->seed, sizeof(unsigned int), 1, f);

    /* Write each tree as a temporary file, then embed */
    for (int t = 0; t < gb->n_estimators; t++) {
        if (!gb->trees[t]) {
            int marker = 0;
            fwrite(&marker, sizeof(int), 1, f);
            continue;
        }
        int marker = 1;
        fwrite(&marker, sizeof(int), 1, f);

        /* Serialize tree to temp file, then read and write */
        char tmppath[256];
        snprintf(tmppath, sizeof(tmppath), "/tmp/gb_tree_%d.bin", t);
        gb->trees[t]->base.save((Estimator*)gb->trees[t], tmppath);

        FILE *tf = fopen(tmppath, "rb");
        if (tf) {
            fseek(tf, 0, SEEK_END);
            long sz = ftell(tf);
            fseek(tf, 0, SEEK_SET);
            fwrite(&sz, sizeof(long), 1, f);
            char *buf = malloc(sz);
            fread(buf, 1, sz, tf);
            fwrite(buf, 1, sz, f);
            free(buf);
            fclose(tf);
            remove(tmppath);
        }
    }

    fclose(f);
    return 0;
}

Estimator* gradient_boosting_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, GB_SUBTYPE) != 0) {
        fclose(f);
        return NULL;
    }

    int n_estimators, max_depth, min_samples_split, n_classes, n_features;
    double learning_rate, subsample, init_prediction;
    unsigned int seed;

    fread(&n_estimators, sizeof(int), 1, f);
    fread(&learning_rate, sizeof(double), 1, f);
    fread(&max_depth, sizeof(int), 1, f);
    fread(&min_samples_split, sizeof(int), 1, f);
    fread(&subsample, sizeof(double), 1, f);
    fread(&init_prediction, sizeof(double), 1, f);
    fread(&n_classes, sizeof(int), 1, f);
    fread(&n_features, sizeof(int), 1, f);
    fread(&seed, sizeof(unsigned int), 1, f);

    GradientBoosting *gb = gradient_boosting_create(
        n_estimators, learning_rate, max_depth, min_samples_split, subsample
    );
    if (!gb) { fclose(f); return NULL; }

    gb->init_prediction = init_prediction;
    gb->n_classes = n_classes;
    gb->n_features = n_features;
    gb->seed = seed;

    gb->trees = calloc(n_estimators, sizeof(DecisionTreeRegressor*));
    if (!gb->trees) { fclose(f); free(gb); return NULL; }

    for (int t = 0; t < n_estimators; t++) {
        int marker;
        fread(&marker, sizeof(int), 1, f);
        if (!marker) continue;

        long sz;
        fread(&sz, sizeof(long), 1, f);
        char *buf = malloc(sz);
        fread(buf, 1, sz, f);

        /* Write to temp and load tree */
        char tmppath[256];
        snprintf(tmppath, sizeof(tmppath), "/tmp/gb_load_tree_%d.bin", t);
        FILE *tf = fopen(tmppath, "wb");
        if (tf) {
            fwrite(buf, 1, sz, tf);
            fclose(tf);
            Estimator *tree_est = gb->trees[t]->base.load(tmppath);
            /* load may return NULL — tree stays NULL which is handled in predict */
            gb->trees[t] = (DecisionTreeRegressor*)tree_est;
            remove(tmppath);
        }
        free(buf);
    }

    fclose(f);
    return (Estimator*)gb;
}
