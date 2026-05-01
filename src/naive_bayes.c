/**
 * @file naive_bayes.c
 * @brief Gaussian Naive Bayes classifier implementation
 */

#include "naive_bayes.h"
#include "cml_error.h"
#include "metrics.h"
#include "cml_serialization.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================
 * Forward declarations for Estimator vtable
 * ============================================ */

static Estimator*  gaussian_nb_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     gaussian_nb_predict(const Estimator *self, const Matrix *X);
static Matrix*     gaussian_nb_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      gaussian_nb_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  gaussian_nb_clone(const Estimator *self);
static void        gaussian_nb_free(Estimator *self);
int         gaussian_nb_save(const Estimator *self, const char *path);
Estimator*  gaussian_nb_load(const char *path);

/* ============================================
 * Public API
 * ============================================ */

GaussianNaiveBayes* gaussian_nb_create(void) {
    GaussianNaiveBayes *nb = calloc(1, sizeof(GaussianNaiveBayes));
    if (!nb) return NULL;

    nb->base.type        = MODEL_NAIVE_BAYES;
    nb->base.task        = TASK_CLASSIFICATION;
    nb->base.is_fitted   = 0;
    nb->base.verbose     = VERBOSE_SILENT;

    nb->base.fit            = gaussian_nb_fit;
    nb->base.predict        = gaussian_nb_predict;
    nb->base.predict_proba  = gaussian_nb_predict_proba_impl;
    nb->base.transform      = NULL;
    nb->base.score          = gaussian_nb_score;
    nb->base.clone          = gaussian_nb_clone;
    nb->base.free           = gaussian_nb_free;
    nb->base.save           = gaussian_nb_save;
    nb->base.load           = gaussian_nb_load;
    nb->base.print_summary  = NULL;

    nb->var_smoothing = 1e-9;

    nb->n_classes       = 0;
    nb->n_features      = 0;
    nb->class_priors     = NULL;
    nb->class_count_    = NULL;
    nb->class_log_prior_ = NULL;
    nb->theta_          = NULL;
    nb->var_            = NULL;
    nb->total_samples_  = 0;

    return nb;
}

/* ============================================
 * Internal helpers
 * ============================================ */

static void nb_free_params(GaussianNaiveBayes *nb) {
    if (!nb) return;
    free(nb->class_priors);
    free(nb->class_count_);
    free(nb->class_log_prior_);
    free(nb->theta_);
    free(nb->var_);
    nb->class_priors      = NULL;
    nb->class_count_      = NULL;
    nb->class_log_prior_  = NULL;
    nb->theta_            = NULL;
    nb->var_              = NULL;
    nb->n_classes         = 0;
    nb->n_features        = 0;
    nb->total_samples_    = 0;
}

/**
 * Compute the log of the Gaussian PDF:
 *   log N(x | mean, var) = -0.5 * log(2π * var) - 0.5 * (x - mean)^2 / var
 */
static double log_gaussian_pdf(double x, double mean, double var) {
    double diff = x - mean;
    return -0.5 * log(2.0 * M_PI * var) - 0.5 * diff * diff / var;
}

/* ============================================
 * fit
 * ============================================ */

static Estimator* gaussian_nb_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    GaussianNaiveBayes *nb = (GaussianNaiveBayes*)self;

    /* Free previous fitted state */
    nb_free_params(nb);

    size_t N = X->rows;
    size_t F = X->cols;

    /* Determine number of classes (labels are integers 0..C-1) */
    int max_class = 0;
    for (size_t i = 0; i < N; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    int C = max_class + 1;

    nb->n_classes      = C;
    nb->n_features     = (int)F;
    nb->total_samples_ = (int)N;

    /* Allocate arrays */
    nb->class_count_     = calloc(C, sizeof(int));
    nb->class_priors     = calloc(C, sizeof(double));
    nb->class_log_prior_ = calloc(C, sizeof(double));
    nb->theta_           = calloc((size_t)C * F, sizeof(double));
    nb->var_             = calloc((size_t)C * F, sizeof(double));

    if (!nb->class_count_ || !nb->class_priors || !nb->class_log_prior_ ||
        !nb->theta_ || !nb->var_) {
        nb_free_params(nb);
        cml_set_error(CML_ERROR_MEMORY, "gaussian_nb_fit: memory allocation failed");
        return NULL;
    }

    /* Count samples per class */
    for (size_t i = 0; i < N; i++) {
        int label = (int)y->data[i];
        nb->class_count_[label]++;
    }

    /* Compute class priors */
    for (int c = 0; c < C; c++) {
        nb->class_priors[c] = (double)nb->class_count_[c] / (double)N;
        nb->class_log_prior_[c] = log(nb->class_priors[c]);
    }

    /* Accumulate sum and sum-of-squares per (class, feature) in one pass */
    /* Use theta_ for sums and var_ for sum-of-squares temporarily */
    for (size_t i = 0; i < N; i++) {
        int c = (int)y->data[i];
        for (size_t j = 0; j < F; j++) {
            double val = X->data[i * F + j];
            nb->theta_[c * F + j] += val;
            nb->var_[c * F + j]   += val * val;
        }
    }

    /* Convert sums to mean and variance */
    for (int c = 0; c < C; c++) {
        int count = nb->class_count_[c];
        if (count == 0) continue;
        for (size_t j = 0; j < F; j++) {
            double sum  = nb->theta_[c * F + j];
            double sum2 = nb->var_[c * F + j];
            double mean = sum / count;
            /* Variance = E[X^2] - E[X]^2 */
            double variance = (sum2 / count) - (mean * mean);
            /* Ensure non-negative (floating point can produce tiny negatives) */
            if (variance < 0.0) variance = 0.0;
            variance += nb->var_smoothing;

            nb->theta_[c * F + j] = mean;
            nb->var_[c * F + j]   = variance;
        }
    }

    nb->base.is_fitted = 1;
    return self;
}

/* ============================================
 * predict
 * ============================================ */

static Matrix* gaussian_nb_predict(const Estimator *self, const Matrix *X) {
    const GaussianNaiveBayes *nb = (const GaussianNaiveBayes*)self;
    if (!nb->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "gaussian_nb_predict: model not fitted");
        return NULL;
    }

    int C = nb->n_classes;
    int F = nb->n_features;
    size_t N = X->rows;

    Matrix *predictions = matrix_alloc(N, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < N; i++) {
        const double *x = &X->data[i * X->cols];
        double best_log_post = -DBL_MAX;
        int best_class = 0;

        for (int c = 0; c < C; c++) {
            double log_post = nb->class_log_prior_[c];
            for (int j = 0; j < F; j++) {
                log_post += log_gaussian_pdf(x[j],
                                             nb->theta_[c * F + j],
                                             nb->var_[c * F + j]);
            }
            if (log_post > best_log_post) {
                best_log_post = log_post;
                best_class = c;
            }
        }
        predictions->data[i] = (double)best_class;
    }

    return predictions;
}

/* ============================================
 * predict_proba
 * ============================================ */

Matrix* gaussian_nb_predict_proba(const Estimator *self, const Matrix *X) {
    return gaussian_nb_predict_proba_impl(self, X);
}

static Matrix* gaussian_nb_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const GaussianNaiveBayes *nb = (const GaussianNaiveBayes*)self;
    if (!nb->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "gaussian_nb_predict_proba: model not fitted");
        return NULL;
    }

    int C = nb->n_classes;
    int F = nb->n_features;
    size_t N = X->rows;

    Matrix *proba = matrix_alloc(N, (size_t)C);
    if (!proba) return NULL;

    double *log_post = malloc(sizeof(double) * C);
    if (!log_post) {
        matrix_free(proba);
        return NULL;
    }

    for (size_t i = 0; i < N; i++) {
        const double *x = &X->data[i * X->cols];

        /* Compute log posteriors */
        for (int c = 0; c < C; c++) {
            log_post[c] = nb->class_log_prior_[c];
            for (int j = 0; j < F; j++) {
                log_post[c] += log_gaussian_pdf(x[j],
                                                nb->theta_[c * F + j],
                                                nb->var_[c * F + j]);
            }
        }

        /* Log-sum-exp for numerical stability */
        double max_log = log_post[0];
        for (int c = 1; c < C; c++) {
            if (log_post[c] > max_log) max_log = log_post[c];
        }

        double sum_exp = 0.0;
        for (int c = 0; c < C; c++) {
            log_post[c] = exp(log_post[c] - max_log);
            sum_exp += log_post[c];
        }

        for (int c = 0; c < C; c++) {
            proba->data[i * C + c] = log_post[c] / sum_exp;
        }
    }

    free(log_post);

    return proba;
}

/* ============================================
 * score (accuracy)
 * ============================================ */

static double gaussian_nb_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

/* ============================================
 * clone (unfitted copy for CV)
 * ============================================ */

static Estimator* gaussian_nb_clone(const Estimator *self) {
    (void)self;
    /* Return an unfitted clone with the same hyperparameters */
    return (Estimator*)gaussian_nb_create();
}

/* ============================================
 * free
 * ============================================ */

static void gaussian_nb_free(Estimator *self) {
    GaussianNaiveBayes *nb = (GaussianNaiveBayes*)self;
    if (nb) {
        nb_free_params(nb);
        free(nb);
    }
}

/* ============================================
 * GaussianNB save/load
 * ============================================ */

int gaussian_nb_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const GaussianNaiveBayes *nb = (const GaussianNaiveBayes*)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_GAUSSIAN_NB) != 0) { fclose(f); return -1; }

    cml_ser_write_int(f, nb->n_classes);
    cml_ser_write_int(f, nb->n_features);
    cml_ser_write_int(f, nb->total_samples_);

    int C = nb->n_classes;
    int F = nb->n_features;
    cml_ser_write_doubles(f, nb->theta_, C * F);
    cml_ser_write_doubles(f, nb->var_, C * F);
    cml_ser_write_doubles(f, nb->class_priors, C);
    cml_ser_write_doubles(f, nb->class_log_prior_, C);
    cml_ser_write_ints(f, nb->class_count_, C);

    fclose(f);
    return 0;
}

Estimator* gaussian_nb_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_GAUSSIAN_NB) != 0) { fclose(f); return NULL; }

    int n_classes, n_features, total_samples;
    if (cml_ser_read_int(f, &n_classes) != 0)    { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)   { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &total_samples) != 0) { fclose(f); return NULL; }

    GaussianNaiveBayes *nb = gaussian_nb_create();
    if (!nb) { fclose(f); return NULL; }

    nb->n_classes      = n_classes;
    nb->n_features     = n_features;
    nb->total_samples_ = total_samples;

    int C = n_classes;
    int F = n_features;

    nb->theta_ = malloc((size_t)(C * F) * sizeof(double));
    nb->var_   = malloc((size_t)(C * F) * sizeof(double));
    nb->class_priors     = malloc((size_t)C * sizeof(double));
    nb->class_log_prior_ = malloc((size_t)C * sizeof(double));
    nb->class_count_     = malloc((size_t)C * sizeof(int));

    if (!nb->theta_ || !nb->var_ || !nb->class_priors || !nb->class_log_prior_ || !nb->class_count_) {
        gaussian_nb_free((Estimator*)nb); fclose(f); return NULL;
    }

    if (cml_ser_read_doubles(f, nb->theta_, C * F) != 0 ||
        cml_ser_read_doubles(f, nb->var_, C * F) != 0 ||
        cml_ser_read_doubles(f, nb->class_priors, C) != 0 ||
        cml_ser_read_doubles(f, nb->class_log_prior_, C) != 0 ||
        cml_ser_read_ints(f, nb->class_count_, C) != 0) {
        gaussian_nb_free((Estimator*)nb); fclose(f); return NULL;
    }

    fclose(f);
    nb->base.is_fitted = 1;
    return (Estimator*)nb;
}

/* ============================================
 * Multinomial Naive Bayes
 * ============================================ */

static Estimator*  multinomial_nb_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     multinomial_nb_predict(const Estimator *self, const Matrix *X);
static Matrix*     multinomial_nb_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      multinomial_nb_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  multinomial_nb_clone(const Estimator *self);
static void        multinomial_nb_free(Estimator *self);
int         multinomial_nb_save(const Estimator *self, const char *path);
Estimator*  multinomial_nb_load(const char *path);

MultinomialNB* multinomial_nb_create(void) {
    MultinomialNB *nb = calloc(1, sizeof(MultinomialNB));
    if (!nb) return NULL;

    nb->base.type        = MODEL_NAIVE_BAYES;
    nb->base.task        = TASK_CLASSIFICATION;
    nb->base.is_fitted   = 0;
    nb->base.verbose     = VERBOSE_SILENT;

    nb->base.fit            = multinomial_nb_fit;
    nb->base.predict        = multinomial_nb_predict;
    nb->base.predict_proba  = multinomial_nb_predict_proba_impl;
    nb->base.transform      = NULL;
    nb->base.score          = multinomial_nb_score;
    nb->base.clone          = multinomial_nb_clone;
    nb->base.free           = multinomial_nb_free;
    nb->base.save           = multinomial_nb_save;
    nb->base.load           = multinomial_nb_load;
    nb->base.print_summary  = NULL;

    nb->alpha      = 1.0;  /* Laplace smoothing */
    nb->log_prior  = NULL;
    nb->theta      = NULL;
    nb->n_classes  = 0;
    nb->n_features = 0;

    return nb;
}

static void multinomial_nb_free_params(MultinomialNB *nb) {
    if (!nb) return;
    if (nb->log_prior) { matrix_free(nb->log_prior); nb->log_prior = NULL; }
    if (nb->theta)     { matrix_free(nb->theta);     nb->theta = NULL; }
    nb->n_classes  = 0;
    nb->n_features = 0;
}

static Estimator* multinomial_nb_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    MultinomialNB *nb = (MultinomialNB *)self;

    if (!X || !y || X->rows != y->rows) {
        cml_set_error(CML_ERROR_INVALID_ARG, "multinomial_nb_fit: invalid input");
        return NULL;
    }

    /* Free previous fitted state */
    multinomial_nb_free_params(nb);

    size_t N = X->rows;
    size_t F = X->cols;

    /* Determine number of classes (labels are integers 0..C-1) */
    int max_class = 0;
    for (size_t i = 0; i < N; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    int C = max_class + 1;

    nb->n_classes  = C;
    nb->n_features = (int)F;

    /* Count class occurrences */
    int *class_count = calloc((size_t)C, sizeof(int));
    if (!class_count) return NULL;

    for (size_t i = 0; i < N; i++) {
        int c = (int)y->data[i];
        class_count[c]++;
    }

    /* Compute log prior */
    nb->log_prior = matrix_alloc((size_t)C, 1);
    if (!nb->log_prior) {
        free(class_count);
        return NULL;
    }
    for (int c = 0; c < C; c++) {
        nb->log_prior->data[c] = log((double)class_count[c] / (double)N);
    }

    /* Compute theta: log P(x_j|c) for each class c and feature j */
    nb->theta = matrix_alloc((size_t)C, F);
    if (!nb->theta) {
        free(class_count);
        multinomial_nb_free_params(nb);
        return NULL;
    }

    /* Accumulate feature sums per class */
    /* theta will hold raw sums first, then convert to log probabilities */
    /* Initialize to alpha (smoothing) */
    for (int c = 0; c < C; c++) {
        for (size_t j = 0; j < F; j++) {
            nb->theta->data[(size_t)c * F + j] = 0.0;
        }
    }

    /* Sum feature values per class */
    for (size_t i = 0; i < N; i++) {
        int c = (int)y->data[i];
        for (size_t j = 0; j < F; j++) {
            nb->theta->data[(size_t)c * F + j] += X->data[i * F + j];
        }
    }

    /* Convert to log probabilities with Laplace smoothing */
    for (int c = 0; c < C; c++) {
        /* Total count for class c (sum of all features) */
        double total = 0.0;
        for (size_t j = 0; j < F; j++) {
            total += nb->theta->data[(size_t)c * F + j];
        }
        /* theta(c,j) = log((sum + alpha) / (total + alpha * n_features)) */
        double denom = total + nb->alpha * (double)F;
        for (size_t j = 0; j < F; j++) {
            double numer = nb->theta->data[(size_t)c * F + j] + nb->alpha;
            nb->theta->data[(size_t)c * F + j] = log(numer / denom);
        }
    }

    free(class_count);
    nb->base.is_fitted = 1;
    return self;
}

static Matrix* multinomial_nb_predict(const Estimator *self, const Matrix *X) {
    const MultinomialNB *nb = (const MultinomialNB *)self;
    if (!nb->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "multinomial_nb_predict: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    int C = nb->n_classes;
    int F = nb->n_features;

    Matrix *pred = matrix_alloc(N, 1);
    if (!pred) return NULL;

    for (size_t i = 0; i < N; i++) {
        double best_score = -1e300;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            double score = nb->log_prior->data[c];
            for (int j = 0; j < F; j++) {
                score += X->data[i * (size_t)F + j] * nb->theta->data[(size_t)c * (size_t)F + j];
            }
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        pred->data[i] = (double)best_class;
    }

    return pred;
}

static Matrix* multinomial_nb_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const MultinomialNB *nb = (const MultinomialNB *)self;
    if (!nb->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "multinomial_nb_predict_proba: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    int C = nb->n_classes;
    int F = nb->n_features;

    Matrix *proba = matrix_alloc(N, (size_t)C);
    if (!proba) return NULL;

    for (size_t i = 0; i < N; i++) {
        /* Compute log joint likelihood per class */
        double max_log = -1e300;
        for (int c = 0; c < C; c++) {
            double log_prob = nb->log_prior->data[c];
            for (int j = 0; j < F; j++) {
                log_prob += X->data[i * (size_t)F + j] * nb->theta->data[(size_t)c * (size_t)F + j];
            }
            proba->data[i * (size_t)C + c] = log_prob;
            if (log_prob > max_log) max_log = log_prob;
        }

        /* Exponentiate with log-sum-exp trick and normalize */
        double sum_exp = 0.0;
        for (int c = 0; c < C; c++) {
            proba->data[i * (size_t)C + c] = exp(proba->data[i * (size_t)C + c] - max_log);
            sum_exp += proba->data[i * (size_t)C + c];
        }
        for (int c = 0; c < C; c++) {
            proba->data[i * (size_t)C + c] /= sum_exp;
        }
    }

    return proba;
}

static double multinomial_nb_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

static Estimator* multinomial_nb_clone(const Estimator *self) {
    (void)self;
    return (Estimator *)multinomial_nb_create();
}

static void multinomial_nb_free(Estimator *self) {
    MultinomialNB *nb = (MultinomialNB *)self;
    if (nb) {
        multinomial_nb_free_params(nb);
        free(nb);
    }
}

/* ============================================
 * MultinomialNB save/load
 * ============================================ */

int multinomial_nb_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const MultinomialNB *nb = (const MultinomialNB *)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_MULTINOMIAL_NB) != 0) { fclose(f); return -1; }

    cml_ser_write_double(f, nb->alpha);
    cml_ser_write_int(f, nb->n_classes);
    cml_ser_write_int(f, nb->n_features);
    cml_ser_write_matrix(f, nb->theta);
    cml_ser_write_matrix(f, nb->log_prior);

    fclose(f);
    return 0;
}

Estimator* multinomial_nb_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_MULTINOMIAL_NB) != 0) { fclose(f); return NULL; }

    double alpha;
    int n_classes, n_features;
    if (cml_ser_read_double(f, &alpha) != 0)    { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_classes) != 0)   { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)  { fclose(f); return NULL; }

    MultinomialNB *nb = multinomial_nb_create();
    if (!nb) { fclose(f); return NULL; }

    nb->alpha      = alpha;
    nb->n_classes  = n_classes;
    nb->n_features = n_features;

    nb->theta = cml_ser_read_matrix(f);
    if (!nb->theta) { multinomial_nb_free((Estimator*)nb); fclose(f); return NULL; }

    nb->log_prior = cml_ser_read_matrix(f);
    if (!nb->log_prior) { multinomial_nb_free((Estimator*)nb); fclose(f); return NULL; }

    fclose(f);
    nb->base.is_fitted = 1;
    return (Estimator*)nb;
}
