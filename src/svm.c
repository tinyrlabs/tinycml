/**
 * @file svm.c
 * @brief Linear SVM classifier implementation using sub-gradient descent
 */

#include "svm.h"
#include "cml_error.h"
#include "metrics.h"
#include "utils.h"
#include "cml_serialization.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Forward declarations for Estimator vtable
 * ============================================ */

static Estimator*  linear_svc_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     linear_svc_predict(const Estimator *self, const Matrix *X);
static Matrix*     linear_svc_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      linear_svc_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  linear_svc_clone(const Estimator *self);
static void        linear_svc_free(Estimator *self);
int         linear_svc_save(const Estimator *self, const char *path);
Estimator*  linear_svc_load(const char *path);

/* ============================================
 * Public API
 * ============================================ */

LinearSVC* linear_svc_create(void) {
    LinearSVC *svc = calloc(1, sizeof(LinearSVC));
    if (!svc) return NULL;

    svc->base.type        = MODEL_SVM;
    svc->base.task        = TASK_CLASSIFICATION;
    svc->base.is_fitted   = 0;
    svc->base.verbose     = VERBOSE_SILENT;

    svc->base.fit            = linear_svc_fit;
    svc->base.predict        = linear_svc_predict;
    svc->base.predict_proba  = linear_svc_predict_proba_impl;
    svc->base.transform      = NULL;
    svc->base.score          = linear_svc_score;
    svc->base.clone          = linear_svc_clone;
    svc->base.free           = linear_svc_free;
    svc->base.save           = linear_svc_save;
    svc->base.load           = linear_svc_load;
    svc->base.print_summary  = NULL;

    /* Default hyperparameters */
    svc->C             = 1.0;
    svc->learning_rate = 0.001;
    svc->max_iter      = 1000;
    svc->tol           = 1e-4;

    /* Fitted parameters */
    svc->weights        = NULL;
    svc->bias           = 0.0;
    svc->n_features     = 0;
    svc->labels_converted = 0;

    return svc;
}

/* ============================================
 * Internal helpers
 * ============================================ */

static void svc_free_params(LinearSVC *svc) {
    if (!svc) return;
    if (svc->weights) {
        matrix_free(svc->weights);
        svc->weights = NULL;
    }
    svc->bias       = 0.0;
    svc->n_features = 0;
    svc->labels_converted = 0;
}

/**
 * Compute dot product w · x_i (row i of X with weight vector)
 */
static double dot_product(const Matrix *w, const Matrix *X, size_t row) {
    double sum = 0.0;
    size_t cols = X->cols;
    for (size_t j = 0; j < cols; j++) {
        sum += w->data[j] * X->data[row * cols + j];
    }
    return sum;
}

/* ============================================
 * fit — Sub-gradient descent with hinge loss
 * ============================================ */

static Estimator* linear_svc_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    LinearSVC *svc = (LinearSVC*)self;

    /* Free previous fitted state */
    svc_free_params(svc);

    size_t N = X->rows;
    size_t F = X->cols;
    svc->n_features = (int)F;

    /* Check if labels are {0, 1} and need conversion to {-1, +1} */
    int needs_conversion = 0;
    for (size_t i = 0; i < N; i++) {
        double label = y->data[i];
        if (label == 0.0) {
            needs_conversion = 1;
            break;
        }
    }
    svc->labels_converted = needs_conversion;

    /* Create label array (convert to {-1, +1} if needed) */
    double *labels = malloc(N * sizeof(double));
    if (!labels) return NULL;

    for (size_t i = 0; i < N; i++) {
        if (needs_conversion) {
            labels[i] = (y->data[i] <= 0.5) ? -1.0 : 1.0;
        } else {
            labels[i] = y->data[i];
        }
    }

    /* Initialize weights to small random values */
    svc->weights = matrix_alloc(F, 1);
    if (!svc->weights) {
        free(labels);
        return NULL;
    }

    rand_seed(42);
    for (size_t j = 0; j < F; j++) {
        svc->weights->data[j] = rand_uniform() * 0.01 - 0.005;  /* small random in [-0.005, 0.005] */
    }
    svc->bias = 0.0;

    double lr = svc->learning_rate;
    double C  = svc->C;
    double prev_loss = 1e18;

    /* Training loop */
    for (int epoch = 0; epoch < svc->max_iter; epoch++) {
        /* Shuffle sample indices for stochastic gradient descent */
        size_t *indices = malloc(N * sizeof(size_t));
        if (!indices) {
            free(labels);
            svc_free_params(svc);
            return NULL;
        }
        for (size_t i = 0; i < N; i++) indices[i] = i;
        shuffle_indices(indices, N);

        double epoch_loss = 0.0;

        for (size_t idx = 0; idx < N; idx++) {
            size_t i = indices[idx];
            double yi = labels[i];
            double decision = yi * (dot_product(svc->weights, X, i) + svc->bias);

            if (decision < 1.0) {
                /* Misclassified or within margin: update w += lr*(C*yi*xi - w), b += lr*C*yi */
                for (size_t j = 0; j < F; j++) {
                    double xij = X->data[i * F + j];
                    svc->weights->data[j] += lr * (C * yi * xij - svc->weights->data[j]);
                }
                svc->bias += lr * C * yi;
                epoch_loss += 1.0 - decision;
            } else {
                /* Correctly classified: L2 decay w -= lr*w */
                for (size_t j = 0; j < F; j++) {
                    svc->weights->data[j] -= lr * svc->weights->data[j];
                }
            }
        }

        /* Add L2 regularization to loss */
        double w_norm_sq = 0.0;
        for (size_t j = 0; j < F; j++) {
            w_norm_sq += svc->weights->data[j] * svc->weights->data[j];
        }
        epoch_loss = epoch_loss / (double)N + 0.5 * w_norm_sq;

        free(indices);

        /* Check convergence */
        if (epoch > 0 && fabs(prev_loss - epoch_loss) < svc->tol) {
            break;
        }
        prev_loss = epoch_loss;
    }

    free(labels);
    svc->base.is_fitted = 1;
    return self;
}

/* ============================================
 * predict — sign(w · x + b)
 * ============================================ */

static Matrix* linear_svc_predict(const Estimator *self, const Matrix *X) {
    const LinearSVC *svc = (const LinearSVC*)self;
    if (!svc->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "linear_svc_predict: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    Matrix *predictions = matrix_alloc(N, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < N; i++) {
        double val = dot_product(svc->weights, X, i) + svc->bias;
        double raw = (val >= 0.0) ? 1.0 : -1.0;

        /* If labels were converted from {0,1}, convert back */
        if (svc->labels_converted) {
            predictions->data[i] = (raw > 0.0) ? 1.0 : 0.0;
        } else {
            predictions->data[i] = raw;
        }
    }

    return predictions;
}

/* ============================================
 * predict_proba — sigmoid on raw margin
 * ============================================ */

static Matrix* linear_svc_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const LinearSVC *svc = (const LinearSVC*)self;
    if (!svc->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "linear_svc_predict_proba: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    Matrix *proba = matrix_alloc(N, 1);
    if (!proba) return NULL;

    for (size_t i = 0; i < N; i++) {
        double margin = dot_product(svc->weights, X, i) + svc->bias;
        /* Sigmoid: P = 1 / (1 + exp(-margin)) */
        /* Clamp to avoid overflow */
        double clamped = margin;
        if (clamped > 500.0) clamped = 500.0;
        if (clamped < -500.0) clamped = -500.0;
        proba->data[i] = 1.0 / (1.0 + exp(-clamped));
    }

    return proba;
}

Matrix* linear_svc_predict_proba(const LinearSVC *model, const Matrix *X) {
    return linear_svc_predict_proba_impl((const Estimator*)model, X);
}

/* ============================================
 * score — accuracy
 * ============================================ */

static double linear_svc_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

/* ============================================
 * clone — unfitted copy
 * ============================================ */

static Estimator* linear_svc_clone(const Estimator *self) {
    (void)self;
    return (Estimator*)linear_svc_create();
}

/* ============================================
 * free
 * ============================================ */

static void linear_svc_free(Estimator *self) {
    LinearSVC *svc = (LinearSVC*)self;
    if (svc) {
        svc_free_params(svc);
        free(svc);
    }
}

void linear_svc_free_impl(Estimator *self) {
    linear_svc_free(self);
}

/* ============================================
 * SVMClassifier — Linear + RBF kernel
 * ============================================ */

/* Forward declarations */
static Estimator*  svm_clf_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     svm_clf_predict(const Estimator *self, const Matrix *X);
static Matrix*     svm_clf_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      svm_clf_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  svm_clf_clone(const Estimator *self);
static void        svm_clf_free(Estimator *self);
static int         svm_clf_save(const Estimator *self, const char *path);
static Estimator*  svm_clf_load(const char *path);

static void svm_clf_free_params(SVMClassifier *svm) {
    if (!svm) return;
    if (svm->weights) { matrix_free(svm->weights); svm->weights = NULL; }
    if (svm->support_vectors) { matrix_free(svm->support_vectors); svm->support_vectors = NULL; }
    if (svm->alphas) { free(svm->alphas); svm->alphas = NULL; }
    if (svm->labels_train) { free(svm->labels_train); svm->labels_train = NULL; }
    svm->bias = 0.0;
    svm->n_support = 0;
    svm->n_features = 0;
    svm->labels_converted = 0;
}

SVMClassifier* svm_classifier_create(SVMKernelType kernel) {
    SVMClassifier *svm = calloc(1, sizeof(SVMClassifier));
    if (!svm) return NULL;

    svm->base.type        = MODEL_SVM;
    svm->base.task        = TASK_CLASSIFICATION;
    svm->base.is_fitted   = 0;
    svm->base.verbose     = VERBOSE_SILENT;

    svm->base.fit            = svm_clf_fit;
    svm->base.predict        = svm_clf_predict;
    svm->base.predict_proba  = svm_clf_predict_proba_impl;
    svm->base.transform      = NULL;
    svm->base.score          = svm_clf_score;
    svm->base.clone          = svm_clf_clone;
    svm->base.free           = svm_clf_free;
    svm->base.save           = svm_clf_save;
    svm->base.load           = svm_clf_load;
    svm->base.print_summary  = NULL;

    svm->kernel    = kernel;
    svm->C         = 1.0;
    svm->gamma     = -1.0;   /* auto */
    svm->lr        = 0.01;
    svm->max_iter  = 1000;
    svm->tol       = 1e-4;

    svm->weights         = NULL;
    svm->bias            = 0.0;
    svm->support_vectors = NULL;
    svm->alphas          = NULL;
    svm->labels_train    = NULL;
    svm->n_support       = 0;
    svm->n_features      = 0;
    svm->labels_converted = 0;

    return svm;
}

/* --- RBF kernel helper --- */
static double rbf_kernel(const double *x1, const double *x2, size_t dim, double gamma) {
    double sum = 0.0;
    for (size_t j = 0; j < dim; j++) {
        double d = x1[j] - x2[j];
        sum += d * d;
    }
    return exp(-gamma * sum);
}

/* --- Fit (linear kernel) --- */
static Estimator* svm_clf_fit_linear(SVMClassifier *svm, const Matrix *X, const Matrix *y) {
    size_t N = X->rows;
    size_t F = X->cols;
    svm->n_features = (int)F;

    /* Check label conversion */
    int needs_conversion = 0;
    for (size_t i = 0; i < N; i++) {
        if (y->data[i] == 0.0) { needs_conversion = 1; break; }
    }
    svm->labels_converted = needs_conversion;

    double *labels = malloc(N * sizeof(double));
    if (!labels) return NULL;
    for (size_t i = 0; i < N; i++) {
        labels[i] = needs_conversion ? ((y->data[i] <= 0.5) ? -1.0 : 1.0) : y->data[i];
    }

    svm->weights = matrix_alloc(F, 1);
    if (!svm->weights) { free(labels); return NULL; }

    rand_seed(42);
    for (size_t j = 0; j < F; j++)
        svm->weights->data[j] = rand_uniform() * 0.01 - 0.005;
    svm->bias = 0.0;

    double lr = svm->lr;
    double C  = svm->C;
    double prev_loss = 1e18;

    for (int epoch = 0; epoch < svm->max_iter; epoch++) {
        size_t *indices = malloc(N * sizeof(size_t));
        if (!indices) { free(labels); svm_clf_free_params(svm); return NULL; }
        for (size_t i = 0; i < N; i++) indices[i] = i;
        shuffle_indices(indices, N);

        double epoch_loss = 0.0;
        for (size_t idx = 0; idx < N; idx++) {
            size_t i = indices[idx];
            double yi = labels[i];
            double decision = yi * (dot_product(svm->weights, X, i) + svm->bias);
            if (decision < 1.0) {
                for (size_t j = 0; j < F; j++) {
                    svm->weights->data[j] += lr * (C * yi * X->data[i * F + j] - svm->weights->data[j]);
                }
                svm->bias += lr * C * yi;
                epoch_loss += 1.0 - decision;
            } else {
                for (size_t j = 0; j < F; j++)
                    svm->weights->data[j] -= lr * svm->weights->data[j];
            }
        }

        double w_norm_sq = 0.0;
        for (size_t j = 0; j < F; j++)
            w_norm_sq += svm->weights->data[j] * svm->weights->data[j];
        epoch_loss = epoch_loss / (double)N + 0.5 * w_norm_sq;

        free(indices);
        if (epoch > 0 && fabs(prev_loss - epoch_loss) < svm->tol) break;
        prev_loss = epoch_loss;
    }

    free(labels);
    svm->base.is_fitted = 1;
    return (Estimator*)svm;
}

/* --- Fit (RBF kernel) --- */
static Estimator* svm_clf_fit_rbf(SVMClassifier *svm, const Matrix *X, const Matrix *y) {
    size_t N = X->rows;
    size_t F = X->cols;
    svm->n_features = (int)F;

    /* Determine gamma */
    double gamma = svm->gamma;
    if (gamma < 0.0) gamma = 1.0 / (double)F;

    /* Convert labels */
    int needs_conversion = 0;
    for (size_t i = 0; i < N; i++) {
        if (y->data[i] == 0.0) { needs_conversion = 1; break; }
    }
    svm->labels_converted = needs_conversion;

    double *labels = malloc(N * sizeof(double));
    if (!labels) return NULL;
    for (size_t i = 0; i < N; i++) {
        labels[i] = needs_conversion ? ((y->data[i] <= 0.5) ? -1.0 : 1.0) : y->data[i];
    }

    /* Pre-compute kernel matrix K[i][j] = exp(-gamma * ||x_i - x_j||^2) */
    double *K = calloc(N * N, sizeof(double));
    if (!K) { free(labels); return NULL; }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double sum = 0.0;
            for (size_t d = 0; d < F; d++) {
                double diff = X->data[i * F + d] - X->data[j * F + d];
                sum += diff * diff;
            }
            double val = exp(-gamma * sum);
            K[i * N + j] = val;
            K[j * N + i] = val;
        }
    }

    /* Initialize dual variables */
    double *alphas = calloc(N, sizeof(double));
    if (!alphas) { free(K); free(labels); return NULL; }

    double bias = 0.0;
    double lr = svm->lr;
    double C  = svm->C;

    /* SGD on dual problem */
    for (int epoch = 0; epoch < svm->max_iter; epoch++) {
        size_t *indices = malloc(N * sizeof(size_t));
        if (!indices) { free(K); free(labels); free(alphas); return NULL; }
        for (size_t i = 0; i < N; i++) indices[i] = i;
        shuffle_indices(indices, N);

        for (size_t idx = 0; idx < N; idx++) {
            size_t i = indices[idx];
            double yi = labels[i];

            /* Compute decision value: f(x_i) = sum_j alpha_j * y_j * K(i,j) + b */
            double decision = 0.0;
            for (size_t j = 0; j < N; j++) {
                decision += alphas[j] * labels[j] * K[i * N + j];
            }
            decision += bias;

            if (yi * decision < 1.0) {
                /* Update alpha_i: alpha_i += lr * (1 - alpha_i / C) clipped to [0, C] */
                double new_alpha = alphas[i] + lr * (1.0 - alphas[i] / C);
                if (new_alpha < 0.0) new_alpha = 0.0;
                if (new_alpha > C) new_alpha = C;
                alphas[i] = new_alpha;
                bias += lr * yi;
            } else {
                /* Shrink alpha slightly toward 0 */
                double new_alpha = alphas[i] - lr * alphas[i] / C;
                if (new_alpha < 0.0) new_alpha = 0.0;
                alphas[i] = new_alpha;
            }
        }
        free(indices);
    }

    /* Identify support vectors (alpha > epsilon) */
    double epsilon = 1e-6;
    int sv_count = 0;
    for (size_t i = 0; i < N; i++) {
        if (alphas[i] > epsilon) sv_count++;
    }
    if (sv_count == 0) sv_count = (int)N;  /* fallback: keep all */

    /* Store support vectors, their alphas, and labels */
    svm->support_vectors = matrix_alloc((size_t)sv_count, F);
    svm->alphas = malloc(sv_count * sizeof(double));
    svm->labels_train = malloc(sv_count * sizeof(double));
    if (!svm->support_vectors || !svm->alphas || !svm->labels_train) {
        free(K); free(labels); free(alphas);
        if (svm->support_vectors) matrix_free(svm->support_vectors);
        if (svm->alphas) free(svm->alphas);
        if (svm->labels_train) free(svm->labels_train);
        return NULL;
    }

    int idx = 0;
    for (size_t i = 0; i < N; i++) {
        if (alphas[i] > epsilon) {
            for (size_t j = 0; j < F; j++) {
                svm->support_vectors->data[idx * F + j] = X->data[i * F + j];
            }
            svm->alphas[idx] = alphas[i];
            svm->labels_train[idx] = labels[i];
            idx++;
        }
    }
    svm->n_support = sv_count;
    svm->bias = bias;

    free(K);
    free(labels);
    free(alphas);
    svm->base.is_fitted = 1;
    return (Estimator*)svm;
}

static Estimator* svm_clf_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    SVMClassifier *svm = (SVMClassifier*)self;
    svm_clf_free_params(svm);

    if (svm->kernel == CML_KERNEL_LINEAR) {
        return svm_clf_fit_linear(svm, X, y);
    } else {
        return svm_clf_fit_rbf(svm, X, y);
    }
}

/* --- predict --- */
static Matrix* svm_clf_predict(const Estimator *self, const Matrix *X) {
    const SVMClassifier *svm = (const SVMClassifier*)self;
    if (!svm->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "svm_clf_predict: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    size_t F = (size_t)svm->n_features;
    Matrix *pred = matrix_alloc(N, 1);
    if (!pred) return NULL;

    if (svm->kernel == CML_KERNEL_LINEAR) {
        for (size_t i = 0; i < N; i++) {
            double val = dot_product(svm->weights, X, i) + svm->bias;
            double raw = (val >= 0.0) ? 1.0 : -1.0;
            if (svm->labels_converted) {
                pred->data[i] = (raw > 0.0) ? 1.0 : 0.0;
            } else {
                pred->data[i] = raw;
            }
        }
    } else {
        /* RBF kernel: compute decision value from support vectors */
        double gamma = svm->gamma < 0.0 ? 1.0 / (double)F : svm->gamma;
        for (size_t i = 0; i < N; i++) {
            double decision = 0.0;
            for (int s = 0; s < svm->n_support; s++) {
                double k = rbf_kernel(&X->data[i * F],
                                      &svm->support_vectors->data[(size_t)s * F],
                                      F, gamma);
                decision += svm->alphas[s] * svm->labels_train[s] * k;
            }
            decision += svm->bias;
            double raw = (decision >= 0.0) ? 1.0 : -1.0;
            if (svm->labels_converted) {
                pred->data[i] = (raw > 0.0) ? 1.0 : 0.0;
            } else {
                pred->data[i] = raw;
            }
        }
    }
    return pred;
}

/* --- predict_proba --- */
static Matrix* svm_clf_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const SVMClassifier *svm = (const SVMClassifier*)self;
    if (!svm->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "svm_clf_predict_proba: model not fitted");
        return NULL;
    }

    size_t N = X->rows;
    size_t F = (size_t)svm->n_features;
    Matrix *proba = matrix_alloc(N, 1);
    if (!proba) return NULL;

    if (svm->kernel == CML_KERNEL_LINEAR) {
        for (size_t i = 0; i < N; i++) {
            double margin = dot_product(svm->weights, X, i) + svm->bias;
            if (margin > 500.0) margin = 500.0;
            if (margin < -500.0) margin = -500.0;
            proba->data[i] = 1.0 / (1.0 + exp(-margin));
        }
    } else {
        double gamma = svm->gamma < 0.0 ? 1.0 / (double)F : svm->gamma;
        for (size_t i = 0; i < N; i++) {
            double decision = 0.0;
            for (int s = 0; s < svm->n_support; s++) {
                double k = rbf_kernel(&X->data[i * F],
                                      &svm->support_vectors->data[(size_t)s * F],
                                      F, gamma);
                decision += svm->alphas[s] * svm->labels_train[s] * k;
            }
            decision += svm->bias;
            if (decision > 500.0) decision = 500.0;
            if (decision < -500.0) decision = -500.0;
            proba->data[i] = 1.0 / (1.0 + exp(-decision));
        }
    }
    return proba;
}

static double svm_clf_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

static Estimator* svm_clf_clone(const Estimator *self) {
    const SVMClassifier *svm = (const SVMClassifier*)self;
    SVMClassifier *copy = svm_classifier_create(svm->kernel);
    if (!copy) return NULL;
    copy->C = svm->C;
    copy->gamma = svm->gamma;
    copy->lr = svm->lr;
    copy->max_iter = svm->max_iter;
    copy->tol = svm->tol;
    return (Estimator*)copy;
}

static void svm_clf_free(Estimator *self) {
    SVMClassifier *svm = (SVMClassifier*)self;
    if (svm) {
        svm_clf_free_params(svm);
        free(svm);
    }
}

void svm_classifier_free_impl(Estimator *self) {
    svm_clf_free(self);
}

int svm_classifier_save(const Estimator *self, const char *path) {
    return svm_clf_save(self, path);
}

Estimator* svm_classifier_load(const char *path) {
    return svm_clf_load(path);
}

/* ============================================
 * LinearSVC save/load
 * ============================================ */

int linear_svc_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const LinearSVC *svc = (const LinearSVC *)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_LINEAR_SVC) != 0) { fclose(f); return -1; }

    cml_ser_write_double(f, svc->C);
    cml_ser_write_double(f, svc->learning_rate);
    cml_ser_write_int(f, svc->max_iter);
    cml_ser_write_double(f, svc->tol);
    cml_ser_write_int(f, svc->n_features);
    cml_ser_write_matrix(f, svc->weights);
    cml_ser_write_double(f, svc->bias);
    cml_ser_write_int(f, svc->labels_converted);

    fclose(f);
    return 0;
}

Estimator* linear_svc_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_LINEAR_SVC) != 0) { fclose(f); return NULL; }

    double C, learning_rate, tol, bias;
    int max_iter, n_features, labels_converted;

    if (cml_ser_read_double(f, &C) != 0)            { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &learning_rate) != 0) { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &max_iter) != 0)         { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &tol) != 0)           { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)       { fclose(f); return NULL; }

    LinearSVC *svc = linear_svc_create();
    if (!svc) { fclose(f); return NULL; }

    svc->C = C;
    svc->learning_rate = learning_rate;
    svc->max_iter = max_iter;
    svc->tol = tol;
    svc->n_features = n_features;

    svc->weights = cml_ser_read_matrix(f);
    if (!svc->weights) { linear_svc_free((Estimator*)svc); fclose(f); return NULL; }

    if (cml_ser_read_double(f, &bias) != 0) { linear_svc_free((Estimator*)svc); fclose(f); return NULL; }
    svc->bias = bias;

    if (cml_ser_read_int(f, &labels_converted) != 0) { linear_svc_free((Estimator*)svc); fclose(f); return NULL; }
    svc->labels_converted = labels_converted;

    fclose(f);
    svc->base.is_fitted = 1;
    return (Estimator*)svc;
}

/* ============================================
 * SVMClassifier save/load
 * ============================================ */

static int svm_clf_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const SVMClassifier *svm = (const SVMClassifier *)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_SVM_CLASSIFIER) != 0) { fclose(f); return -1; }

    cml_ser_write_int(f, (int)svm->kernel);
    cml_ser_write_double(f, svm->C);
    cml_ser_write_double(f, svm->gamma);
    cml_ser_write_int(f, svm->n_support);
    cml_ser_write_int(f, svm->n_features);
    cml_ser_write_int(f, svm->labels_converted);

    if (svm->kernel == CML_KERNEL_LINEAR) {
        /* Linear: weights + bias */
        cml_ser_write_matrix(f, svm->weights);
        cml_ser_write_double(f, svm->bias);
    } else {
        /* RBF: support_vectors + alphas + labels_train + bias */
        cml_ser_write_matrix(f, svm->support_vectors);
        cml_ser_write_doubles(f, svm->alphas, svm->n_support);
        cml_ser_write_doubles(f, svm->labels_train, svm->n_support);
        cml_ser_write_double(f, svm->bias);
    }

    fclose(f);
    return 0;
}

static Estimator* svm_clf_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_SVM_CLASSIFIER) != 0) { fclose(f); return NULL; }

    int kernel_int, n_support, n_features, labels_converted;
    double C, gamma;

    if (cml_ser_read_int(f, &kernel_int) != 0)      { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &C) != 0)             { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &gamma) != 0)          { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_support) != 0)         { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)        { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &labels_converted) != 0)  { fclose(f); return NULL; }

    SVMClassifier *svm = svm_classifier_create((SVMKernelType)kernel_int);
    if (!svm) { fclose(f); return NULL; }

    svm->C = C;
    svm->gamma = gamma;
    svm->n_support = n_support;
    svm->n_features = n_features;
    svm->labels_converted = labels_converted;

    if (svm->kernel == CML_KERNEL_LINEAR) {
        svm->weights = cml_ser_read_matrix(f);
        if (!svm->weights) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }
        if (cml_ser_read_double(f, &svm->bias) != 0) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }
    } else {
        svm->support_vectors = cml_ser_read_matrix(f);
        if (!svm->support_vectors) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }

        svm->alphas = malloc((size_t)n_support * sizeof(double));
        if (!svm->alphas) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }
        if (cml_ser_read_doubles(f, svm->alphas, n_support) != 0) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }

        svm->labels_train = malloc((size_t)n_support * sizeof(double));
        if (!svm->labels_train) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }
        if (cml_ser_read_doubles(f, svm->labels_train, n_support) != 0) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }

        if (cml_ser_read_double(f, &svm->bias) != 0) { svm_clf_free((Estimator*)svm); fclose(f); return NULL; }
    }

    fclose(f);
    svm->base.is_fitted = 1;
    return (Estimator*)svm;
}
