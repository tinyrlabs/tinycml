/**
 * @file ridge.c
 * @brief Ridge (L2-regularized) regression – closed-form solution
 *
 * w = (X^T X + alpha * I)^{-1} X^T y
 * Bias is handled by centering X and y, fitting, then restoring.
 */

#include "ridge.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------------
 * Gauss-Jordan matrix inversion (duplicates the one in linear_regression.c
 * but we keep it static so ridge.c is self-contained).
 * ---------------------------------------------------------------- */
static Matrix* mat_inv(const Matrix *m) {
    if (!m || m->rows != m->cols) return NULL;

    size_t n = m->rows;
    Matrix *aug = matrix_alloc(n, 2 * n);
    if (!aug) return NULL;

    /* [A | I] */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            aug->data[i * 2 * n + j] = m->data[i * n + j];
        }
        aug->data[i * 2 * n + n + i] = 1.0;
    }

    for (size_t col = 0; col < n; col++) {
        /* partial pivot */
        size_t best = col;
        double best_val = fabs(aug->data[col * 2 * n + col]);
        for (size_t row = col + 1; row < n; row++) {
            double v = fabs(aug->data[row * 2 * n + col]);
            if (v > best_val) { best_val = v; best = row; }
        }
        if (best_val < 1e-12) { matrix_free(aug); return NULL; }

        if (best != col) {
            for (size_t j = 0; j < 2 * n; j++) {
                double tmp = aug->data[col * 2 * n + j];
                aug->data[col * 2 * n + j] = aug->data[best * 2 * n + j];
                aug->data[best * 2 * n + j] = tmp;
            }
        }

        double pivot = aug->data[col * 2 * n + col];
        for (size_t j = 0; j < 2 * n; j++)
            aug->data[col * 2 * n + j] /= pivot;

        for (size_t row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug->data[row * 2 * n + col];
            for (size_t j = 0; j < 2 * n; j++)
                aug->data[row * 2 * n + j] -= factor * aug->data[col * 2 * n + j];
        }
    }

    Matrix *inv = matrix_alloc(n, n);
    if (!inv) { matrix_free(aug); return NULL; }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            inv->data[i * n + j] = aug->data[i * 2 * n + n + j];

    matrix_free(aug);
    return inv;
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

RidgeModel* ridge_model_create(void) {
    RidgeModel *m = calloc(1, sizeof(RidgeModel));
    if (!m) return NULL;

    m->base.type  = MODEL_LINEAR_REGRESSION;   /* closest match */
    m->base.task  = TASK_REGRESSION;
    m->base.is_fitted = 0;
    m->base.verbose   = VERBOSE_SILENT;

    m->base.fit        = ridge_fit;
    m->base.predict    = ridge_predict;
    m->base.predict_proba = NULL;
    m->base.transform  = NULL;
    m->base.score      = ridge_score;
    m->base.clone      = ridge_clone;
    m->base.free       = ridge_free;
    m->base.save       = NULL;
    m->base.load       = NULL;
    m->base.print_summary = NULL;

    m->alpha      = 1.0;
    m->max_iter   = 0;
    m->tol        = 0.0;
    m->weights    = NULL;
    m->bias       = 0.0;
    m->n_features = 0;

    return m;
}

Estimator* ridge_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return NULL;

    RidgeModel *m = (RidgeModel *)self;

    size_t n = X->rows;
    size_t p = X->cols;

    /* Free previous fit */
    if (m->weights) { matrix_free(m->weights); m->weights = NULL; }

    /* Center X and y to handle intercept cleanly */
    double *x_mean = calloc(p, sizeof(double));
    double  y_mean = 0.0;
    if (!x_mean) return NULL;

    for (size_t i = 0; i < n; i++) {
        y_mean += y->data[i];
        for (size_t j = 0; j < p; j++)
            x_mean[j] += X->data[i * p + j];
    }
    y_mean /= (double)n;
    for (size_t j = 0; j < p; j++)
        x_mean[j] /= (double)n;

    /* Build centered Xc and yc */
    Matrix *Xc = matrix_alloc(n, p);
    Matrix *yc = matrix_alloc(n, 1);
    if (!Xc || !yc) {
        free(x_mean);
        matrix_free(Xc); matrix_free(yc);
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        yc->data[i] = y->data[i] - y_mean;
        for (size_t j = 0; j < p; j++)
            Xc->data[i * p + j] = X->data[i * p + j] - x_mean[j];
    }

    /* Xc^T */
    Matrix *Xct = matrix_transpose(Xc);
    if (!Xct) {
        free(x_mean); matrix_free(Xc); matrix_free(yc);
        return NULL;
    }

    /* A = Xc^T Xc */
    Matrix *A = matrix_matmul(Xct, Xc);
    matrix_free(Xct);
    if (!A) { free(x_mean); matrix_free(Xc); matrix_free(yc); return NULL; }

    /* Add alpha to diagonal */
    for (size_t i = 0; i < p; i++)
        A->data[i * p + i] += m->alpha;

    /* A_inv */
    Matrix *A_inv = mat_inv(A);
    matrix_free(A);
    if (!A_inv) { free(x_mean); matrix_free(Xc); matrix_free(yc); return NULL; }

    /* Xc^T yc */
    Matrix *Xct2 = matrix_transpose(Xc);
    matrix_free(Xc);
    if (!Xct2) { free(x_mean); matrix_free(yc); matrix_free(A_inv); return NULL; }

    Matrix *Xty = matrix_matmul(Xct2, yc);
    matrix_free(Xct2);
    matrix_free(yc);
    if (!Xty) { free(x_mean); matrix_free(A_inv); return NULL; }

    /* w = A_inv * Xty */
    m->weights = matrix_matmul(A_inv, Xty);
    matrix_free(A_inv);
    matrix_free(Xty);
    if (!m->weights) { free(x_mean); return NULL; }

    /* bias = y_mean - x_mean^T w */
    m->bias = y_mean;
    for (size_t j = 0; j < p; j++)
        m->bias -= x_mean[j] * m->weights->data[j];
    free(x_mean);

    m->n_features = (int)p;
    m->base.is_fitted = 1;
    return self;
}

Matrix* ridge_predict(const Estimator *self, const Matrix *X) {
    if (!estimator_check_fitted(self, "predict")) return NULL;

    const RidgeModel *m = (const RidgeModel *)self;
    size_t n = X->rows;
    size_t p = (size_t)m->n_features;

    Matrix *pred = matrix_alloc(n, 1);
    if (!pred) return NULL;

    for (size_t i = 0; i < n; i++) {
        double val = m->bias;
        for (size_t j = 0; j < p; j++)
            val += X->data[i * p + j] * m->weights->data[j];
        pred->data[i] = val;
    }
    return pred;
}

double ridge_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* ridge_clone(const Estimator *self) {
    if (!self) return NULL;
    const RidgeModel *m = (const RidgeModel *)self;
    RidgeModel *c = ridge_model_create();
    if (!c) return NULL;
    c->alpha = m->alpha;
    return (Estimator *)c;
}

void ridge_free(Estimator *self) {
    if (!self) return;
    RidgeModel *m = (RidgeModel *)self;
    matrix_free(m->weights);
    training_history_free(m->base.history);
    free(m);
}
