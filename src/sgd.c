/**
 * sgd.c - Stochastic Gradient Descent classifier and regressor
 *
 * Linear model trained with mini-batch SGD. Supports:
 * - Squared loss (regression)
 * - Log loss (binary classification)
 * - Hinge loss (SVM-style)
 * - Huber loss (robust regression)
 */

#include "sgd.h"
#include "cml_serialization.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── helpers ─────────────────────────────────────────────────────── */

static double sgd_sigmoid(double x) {
    if (x > 500.0) return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

static unsigned int sgd_rand(unsigned int *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Fisher-Yates shuffle */
static void shuffle_indices(size_t *arr, size_t n, unsigned int *seed) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = sgd_rand(seed) % (i + 1);
        size_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* ── create / free ───────────────────────────────────────────────── */

SGDModel* sgd_create(SGDLossType loss, double alpha, double lr, int max_iter) {
    SGDModel *m = calloc(1, sizeof(SGDModel));
    if (!m) return NULL;

    m->loss          = loss;
    m->alpha         = alpha > 0 ? alpha : 0.0001;
    m->learning_rate = lr > 0 ? lr : 0.01;
    m->eta0          = m->learning_rate;
    m->max_iter      = max_iter > 0 ? max_iter : 1000;
    m->tol           = 1e-6;
    m->shuffle       = 1;
    m->seed          = (unsigned int)time(NULL) ^ 0x5DDBEEF;

    m->weights       = NULL;
    m->bias          = 0.0;
    m->n_features    = 0;
    m->n_classes     = 0;
    m->converged     = 0;

    m->base.fit          = sgd_fit;
    m->base.predict      = sgd_predict;
    m->base.predict_proba = sgd_predict_proba;
    m->base.score        = sgd_score;
    m->base.clone        = sgd_clone;
    m->base.free         = sgd_free;
    m->base.type         = MODEL_LINEAR_REGRESSION;

    return m;
}

SGDModel* sgd_classifier_create(double alpha, double lr, int max_iter) {
    return sgd_create(SGD_LOSS_LOG, alpha, lr, max_iter);
}

SGDModel* sgd_regressor_create(double alpha, double lr, int max_iter) {
    return sgd_create(SGD_LOSS_SQUARED, alpha, lr, max_iter);
}

void sgd_free(Estimator *self) {
    if (!self) return;
    SGDModel *m = (SGDModel*)self;
    free(m->weights);
    free(m);
}

/* ── fit ─────────────────────────────────────────────────────────── */

Estimator* sgd_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y || X->rows != y->rows) return NULL;

    SGDModel *m = (SGDModel*)self;
    size_t n = X->rows;
    size_t d = X->cols;
    m->n_features = (int)d;

    /* Determine classes */
    int max_label = 0;
    for (size_t i = 0; i < n; i++) {
        int lbl = (int)y->data[i];
        if (lbl > max_label) max_label = lbl;
    }
    m->n_classes = (m->loss == SGD_LOSS_SQUARED || m->loss == SGD_LOSS_HUBER)
                   ? 1 : max_label + 1;

    /* Initialize weights */
    m->weights = calloc(d, sizeof(double));
    if (!m->weights) return NULL;
    m->bias = 0.0;

    /* Allocate index array for shuffling */
    size_t *indices = malloc(n * sizeof(size_t));
    if (!indices) return NULL;
    for (size_t i = 0; i < n; i++) indices[i] = i;

    double prev_loss = 1e18;

    for (int epoch = 0; epoch < m->max_iter; epoch++) {
        if (m->shuffle) {
            shuffle_indices(indices, n, &m->seed);
        }

        double epoch_loss = 0.0;

        /* Learning rate schedule: inverse scaling */
        double lr = m->eta0 / (1.0 + m->alpha * epoch);

        for (size_t idx = 0; idx < n; idx++) {
            size_t i = indices[idx];

            /* Compute prediction: z = w^T x + b */
            double z = m->bias;
            for (size_t j = 0; j < d; j++) {
                z += m->weights[j] * matrix_get(X, i, j);
            }

            double yi = y->data[i];
            double grad = 0.0;

            switch (m->loss) {
                case SGD_LOSS_SQUARED: {
                    /* grad = (pred - y) */
                    grad = z - yi;
                    epoch_loss += grad * grad;
                    break;
                }
                case SGD_LOSS_LOG: {
                    /* Binary: grad = sigmoid(z) - y */
                    double prob = sgd_sigmoid(z);
                    grad = prob - yi;
                    double loss_val = (yi > 0.5) ? -log(prob + 1e-15) : -log(1.0 - prob + 1e-15);
                    epoch_loss += loss_val;
                    break;
                }
                case SGD_LOSS_HINGE: {
                    /* max(0, 1 - y*z) where y in {-1, +1} */
                    double y_pm = (yi > 0.5) ? 1.0 : -1.0;
                    double margin = y_pm * z;
                    if (margin < 1.0) {
                        grad = -y_pm;
                        epoch_loss += 1.0 - margin;
                    }
                    break;
                }
                case SGD_LOSS_HUBER: {
                    double diff = z - yi;
                    double delta = 1.0;
                    if (fabs(diff) <= delta) {
                        grad = diff;
                        epoch_loss += 0.5 * diff * diff;
                    } else {
                        grad = (diff > 0 ? delta : -delta);
                        epoch_loss += delta * (fabs(diff) - 0.5 * delta);
                    }
                    break;
                }
            }

            /* Update weights: w -= lr * (grad + alpha * w) */
            for (size_t j = 0; j < d; j++) {
                double xij = matrix_get(X, i, j);
                m->weights[j] -= lr * (grad * xij + m->alpha * m->weights[j]);
            }
            m->bias -= lr * grad;
        }

        epoch_loss /= (double)n;

        /* Early stopping */
        if (m->tol > 0 && fabs(prev_loss - epoch_loss) < m->tol) {
            m->converged = 1;
            break;
        }
        prev_loss = epoch_loss;
    }

    free(indices);
    return self;
}

/* ── predict ─────────────────────────────────────────────────────── */

Matrix* sgd_predict(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    SGDModel *m = (SGDModel*)self;

    size_t n = X->rows;
    Matrix *out = matrix_alloc(n, 1);
    if (!out) return NULL;

    int is_cls = (m->loss == SGD_LOSS_LOG || m->loss == SGD_LOSS_HINGE);

    for (size_t i = 0; i < n; i++) {
        double z = m->bias;
        for (size_t j = 0; j < (size_t)m->n_features; j++) {
            z += m->weights[j] * matrix_get(X, i, j);
        }

        if (is_cls) {
            out->data[i] = sgd_sigmoid(z) >= 0.5 ? 1.0 : 0.0;
        } else {
            out->data[i] = z;
        }
    }

    return out;
}

/* ── predict_proba ───────────────────────────────────────────────── */

Matrix* sgd_predict_proba(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    SGDModel *m = (SGDModel*)self;

    int is_cls = (m->loss == SGD_LOSS_LOG || m->loss == SGD_LOSS_HINGE);
    if (!is_cls) return sgd_predict(self, X);

    size_t n = X->rows;
    Matrix *out = matrix_alloc(n, 2);
    if (!out) return NULL;

    for (size_t i = 0; i < n; i++) {
        double z = m->bias;
        for (size_t j = 0; j < (size_t)m->n_features; j++) {
            z += m->weights[j] * matrix_get(X, i, j);
        }
        double p1 = sgd_sigmoid(z);
        matrix_set(out, i, 0, 1.0 - p1);
        matrix_set(out, i, 1, p1);
    }

    return out;
}

/* ── score ───────────────────────────────────────────────────────── */

double sgd_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return -1.0;
    SGDModel *m = (SGDModel*)self;

    Matrix *pred = sgd_predict(self, X);
    if (!pred) return -1.0;

    int is_cls = (m->loss == SGD_LOSS_LOG || m->loss == SGD_LOSS_HINGE);

    double result;
    if (is_cls) {
        size_t correct = 0;
        for (size_t i = 0; i < y->rows; i++) {
            if ((int)(pred->data[i] + 0.5) == (int)(y->data[i] + 0.5)) correct++;
        }
        result = (double)correct / (double)y->rows;
    } else {
        double mean_y = 0.0;
        for (size_t i = 0; i < y->rows; i++) mean_y += y->data[i];
        mean_y /= (double)y->rows;
        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i = 0; i < y->rows; i++) {
            double d = y->data[i] - pred->data[i];
            ss_res += d * d;
            double dm = y->data[i] - mean_y;
            ss_tot += dm * dm;
        }
        result = (ss_tot == 0.0) ? 0.0 : 1.0 - ss_res / ss_tot;
    }

    matrix_free(pred);
    return result;
}

/* ── clone ───────────────────────────────────────────────────────── */

Estimator* sgd_clone(const Estimator *self) {
    if (!self) return NULL;
    SGDModel *src = (SGDModel*)self;

    SGDModel *dst = sgd_create(src->loss, src->alpha, src->eta0, src->max_iter);
    if (!dst) return NULL;

    dst->n_features = src->n_features;
    dst->n_classes = src->n_classes;
    dst->bias = src->bias;
    dst->tol = src->tol;
    dst->converged = src->converged;

    if (src->weights) {
        dst->weights = malloc(src->n_features * sizeof(double));
        if (dst->weights) {
            memcpy(dst->weights, src->weights, src->n_features * sizeof(double));
        }
    }

    return (Estimator*)dst;
}

/* ── save / load ─────────────────────────────────────────────────── */

#define SGD_SUBTYPE 11

int sgd_save(const Estimator *self, const char *path) {
    if (!self || !path) return -1;
    SGDModel *m = (SGDModel*)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, SGD_SUBTYPE) != 0) { fclose(f); return -1; }

    fwrite(&m->loss, sizeof(SGDLossType), 1, f);
    fwrite(&m->alpha, sizeof(double), 1, f);
    fwrite(&m->eta0, sizeof(double), 1, f);
    fwrite(&m->max_iter, sizeof(int), 1, f);
    fwrite(&m->tol, sizeof(double), 1, f);
    fwrite(&m->n_features, sizeof(int), 1, f);
    fwrite(&m->n_classes, sizeof(int), 1, f);
    fwrite(&m->bias, sizeof(double), 1, f);
    fwrite(&m->converged, sizeof(int), 1, f);
    if (m->weights) {
        fwrite(m->weights, sizeof(double), m->n_features, f);
    }

    fclose(f);
    return 0;
}

Estimator* sgd_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, SGD_SUBTYPE) != 0) { fclose(f); return NULL; }

    SGDLossType loss;
    double alpha, eta0, tol, bias;
    int max_iter, n_features, n_classes, converged;

    fread(&loss, sizeof(SGDLossType), 1, f);
    fread(&alpha, sizeof(double), 1, f);
    fread(&eta0, sizeof(double), 1, f);
    fread(&max_iter, sizeof(int), 1, f);
    fread(&tol, sizeof(double), 1, f);
    fread(&n_features, sizeof(int), 1, f);
    fread(&n_classes, sizeof(int), 1, f);
    fread(&bias, sizeof(double), 1, f);
    fread(&converged, sizeof(int), 1, f);

    SGDModel *m = sgd_create(loss, alpha, eta0, max_iter);
    if (!m) { fclose(f); return NULL; }

    m->tol = tol;
    m->n_features = n_features;
    m->n_classes = n_classes;
    m->bias = bias;
    m->converged = converged;

    m->weights = malloc(n_features * sizeof(double));
    if (m->weights) {
        fread(m->weights, sizeof(double), n_features, f);
    }

    fclose(f);
    return (Estimator*)m;
}
