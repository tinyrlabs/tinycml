/**
 * @file lasso.c
 * @brief Lasso (L1-regularized) regression – coordinate descent
 */

#include "lasso.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static double soft_threshold(double rho, double alpha) {
    if (rho > alpha)  return rho - alpha;
    if (rho < -alpha) return rho + alpha;
    return 0.0;
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

LassoModel* lasso_model_create(void) {
    LassoModel *m = calloc(1, sizeof(LassoModel));
    if (!m) return NULL;

    m->base.type  = MODEL_LINEAR_REGRESSION;
    m->base.task  = TASK_REGRESSION;
    m->base.is_fitted = 0;
    m->base.verbose   = VERBOSE_SILENT;

    m->base.fit        = lasso_fit;
    m->base.predict    = lasso_predict;
    m->base.predict_proba = NULL;
    m->base.transform  = NULL;
    m->base.score      = lasso_score;
    m->base.clone      = lasso_clone;
    m->base.free       = lasso_free;
    m->base.save       = NULL;
    m->base.load       = NULL;
    m->base.print_summary = NULL;

    m->alpha      = 1.0;
    m->max_iter   = 1000;
    m->tol        = 1e-6;
    m->weights    = NULL;
    m->bias       = 0.0;
    m->n_features = 0;

    return m;
}

Estimator* lasso_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return NULL;

    LassoModel *m = (LassoModel *)self;
    size_t n = X->rows;
    size_t p = X->cols;

    /* Free previous fit */
    if (m->weights) { matrix_free(m->weights); m->weights = NULL; }

    /* Center X and y */
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

    /* Centered data */
    double *Xc = malloc(n * p * sizeof(double));
    double *yc = malloc(n * sizeof(double));
    if (!Xc || !yc) {
        free(x_mean); free(Xc); free(yc);
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        yc[i] = y->data[i] - y_mean;
        for (size_t j = 0; j < p; j++)
            Xc[i * p + j] = X->data[i * p + j] - x_mean[j];
    }

    /* Pre-compute X_j^T X_j / n for each feature */
    double *xj_sq = calloc(p, sizeof(double));
    if (!xj_sq) {
        free(x_mean); free(Xc); free(yc);
        return NULL;
    }
    for (size_t j = 0; j < p; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double v = Xc[i * p + j];
            sum += v * v;
        }
        xj_sq[j] = sum / (double)n;
    }

    /* Initialize weights to 0 */
    double *w = calloc(p, sizeof(double));
    if (!w) {
        free(x_mean); free(Xc); free(yc); free(xj_sq);
        return NULL;
    }

    /* Residuals r = yc - Xc * w (initially = yc since w=0) */
    double *r = malloc(n * sizeof(double));
    if (!r) {
        free(x_mean); free(Xc); free(yc); free(xj_sq); free(w);
        return NULL;
    }
    memcpy(r, yc, n * sizeof(double));

    /* Coordinate descent */
    for (int iter = 0; iter < m->max_iter; iter++) {
        double max_change = 0.0;

        for (size_t j = 0; j < p; j++) {
            /* Add back contribution of w_j to residual */
            for (size_t i = 0; i < n; i++)
                r[i] += Xc[i * p + j] * w[j];

            /* Compute rho_j = X_j^T r / n */
            double rho = 0.0;
            for (size_t i = 0; i < n; i++)
                rho += Xc[i * p + j] * r[i];
            rho /= (double)n;

            /* Soft threshold */
            double w_new;
            if (xj_sq[j] < 1e-12) {
                w_new = 0.0;
            } else {
                w_new = soft_threshold(rho, m->alpha) / xj_sq[j];
            }

            double change = fabs(w_new - w[j]);
            if (change > max_change) max_change = change;

            /* Update residual */
            for (size_t i = 0; i < n; i++)
                r[i] -= Xc[i * p + j] * w_new;

            w[j] = w_new;
        }

        if (max_change < m->tol) break;
    }

    /* Build weights matrix */
    m->weights = matrix_alloc(p, 1);
    if (!m->weights) {
        free(x_mean); free(Xc); free(yc); free(xj_sq); free(w); free(r);
        return NULL;
    }
    for (size_t j = 0; j < p; j++)
        m->weights->data[j] = w[j];

    /* Compute bias = y_mean - x_mean^T w */
    m->bias = y_mean;
    for (size_t j = 0; j < p; j++)
        m->bias -= x_mean[j] * w[j];

    m->n_features = (int)p;
    m->base.is_fitted = 1;

    free(x_mean); free(Xc); free(yc); free(xj_sq); free(w); free(r);
    return self;
}

Matrix* lasso_predict(const Estimator *self, const Matrix *X) {
    if (!estimator_check_fitted(self, "predict")) return NULL;

    const LassoModel *m = (const LassoModel *)self;
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

double lasso_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* lasso_clone(const Estimator *self) {
    if (!self) return NULL;
    const LassoModel *m = (const LassoModel *)self;
    LassoModel *c = lasso_model_create();
    if (!c) return NULL;
    c->alpha    = m->alpha;
    c->max_iter = m->max_iter;
    c->tol      = m->tol;
    return (Estimator *)c;
}

void lasso_free(Estimator *self) {
    if (!self) return;
    LassoModel *m = (LassoModel *)self;
    matrix_free(m->weights);
    training_history_free(m->base.history);
    free(m);
}
