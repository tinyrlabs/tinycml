/**
 * @file logistic_regression.c
 * @brief Implementation of logistic regression
 */

#include "logistic_regression.h"
#include "cml_error.h"
#include "cml_serialization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================
 * Forward declarations for Estimator vtable
 * ============================================ */

static Estimator*  logreg_model_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     logreg_model_predict(const Estimator *self, const Matrix *X);
static Matrix*     logreg_model_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      logreg_model_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  logreg_model_clone(const Estimator *self);
static void        logreg_model_free(Estimator *self);
int         logreg_model_save(const Estimator *self, const char *path);
Estimator*  logreg_model_load(const char *path);

double sigmoid(double x) {
    /* Clip to prevent overflow */
    if (x > 500.0) return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs) {
    if (!X || !y || X->rows != y->rows || lr <= 0 || epochs <= 0) {
        return NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;

    /* Initialize weights to zero */
    Matrix *weights = matrix_alloc(m, 1);
    if (!weights) {
        return NULL;
    }

    Matrix *Xt = matrix_transpose(X);
    if (!Xt) {
        matrix_free(weights);
        return NULL;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Compute linear combination: z = X * w */
        Matrix *z = matrix_matmul(X, weights);
        if (!z) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Apply sigmoid and compute error */
        Matrix *pred = matrix_alloc(n, 1);
        if (!pred) {
            matrix_free(z);
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) {
            pred->data[i] = sigmoid(z->data[i]);
        }
        matrix_free(z);

        /* Compute error: pred - y */
        Matrix *error = matrix_sub(pred, y);
        matrix_free(pred);
        if (!error) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Compute gradient: X' * error / n */
        Matrix *gradient = matrix_matmul(Xt, error);
        matrix_free(error);
        if (!gradient) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Update weights: w = w - lr * gradient / n */
        for (size_t i = 0; i < m; i++) {
            weights->data[i] -= lr * gradient->data[i] / (double)n;
        }

        matrix_free(gradient);
    }

    matrix_free(Xt);
    return weights;
}

Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights) {
    if (!X || !weights || X->cols != weights->rows) {
        return NULL;
    }

    Matrix *z = matrix_matmul(X, weights);
    if (!z) {
        return NULL;
    }

    Matrix *proba = matrix_alloc(X->rows, 1);
    if (!proba) {
        matrix_free(z);
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        proba->data[i] = sigmoid(z->data[i]);
    }

    matrix_free(z);
    return proba;
}

Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold) {
    Matrix *proba = logreg_predict_proba(X, weights);
    if (!proba) {
        return NULL;
    }

    Matrix *labels = matrix_alloc(X->rows, 1);
    if (!labels) {
        matrix_free(proba);
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        labels->data[i] = proba->data[i] >= threshold ? 1.0 : 0.0;
    }

    matrix_free(proba);
    return labels;
}

/* ============================================
 * Estimator-compatible API implementation
 * ============================================ */

LogisticRegressionModel* logreg_model_create(void) {
    LogisticRegressionModel *model = calloc(1, sizeof(LogisticRegressionModel));
    if (!model) return NULL;

    model->base.type        = MODEL_LOGISTIC_REGRESSION;
    model->base.task        = TASK_CLASSIFICATION;
    model->base.is_fitted   = 0;
    model->base.verbose     = VERBOSE_SILENT;

    model->base.fit            = logreg_model_fit;
    model->base.predict        = logreg_model_predict;
    model->base.predict_proba  = logreg_model_predict_proba_impl;
    model->base.transform      = NULL;
    model->base.score          = logreg_model_score;
    model->base.clone          = logreg_model_clone;
    model->base.free           = logreg_model_free;
    model->base.save           = logreg_model_save;
    model->base.load           = logreg_model_load;
    model->base.print_summary  = NULL;

    /* Default hyperparameters */
    model->learning_rate = 0.1;
    model->max_iter      = 1000;
    model->lambda_val    = 0.0;
    model->tol           = 1e-6;

    model->weights    = NULL;
    model->n_features = 0;
    model->n_classes  = 2;

    return model;
}

static Estimator* logreg_model_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    LogisticRegressionModel *model = (LogisticRegressionModel *)self;

    if (!X || !y || X->rows != y->rows) {
        return NULL;
    }

    /* Free previous weights if re-fitting */
    if (model->weights) {
        matrix_free(model->weights);
        model->weights = NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;

    /* Initialize weights to zero */
    Matrix *weights = matrix_alloc(m, 1);
    if (!weights) {
        return NULL;
    }

    Matrix *Xt = matrix_transpose(X);
    if (!Xt) {
        matrix_free(weights);
        return NULL;
    }

    double lr = model->learning_rate;
    int epochs = model->max_iter;
    double lambda = model->lambda_val;

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* z = X * w */
        Matrix *z = matrix_matmul(X, weights);
        if (!z) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Apply sigmoid */
        Matrix *pred = matrix_alloc(n, 1);
        if (!pred) {
            matrix_free(z);
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) {
            pred->data[i] = sigmoid(z->data[i]);
        }
        matrix_free(z);

        /* error = pred - y */
        Matrix *error = matrix_sub(pred, y);
        matrix_free(pred);
        if (!error) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* gradient = X' * error / n */
        Matrix *gradient = matrix_matmul(Xt, error);
        matrix_free(error);
        if (!gradient) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        /* Update: w = w - lr * (gradient/n + lambda * w) */
        for (size_t i = 0; i < m; i++) {
            double grad = gradient->data[i] / (double)n;
            if (lambda > 0.0) {
                grad += lambda * weights->data[i];
            }
            weights->data[i] -= lr * grad;
        }

        matrix_free(gradient);
    }

    matrix_free(Xt);

    model->weights    = weights;
    model->n_features = (int)m;
    model->n_classes  = 2;
    model->base.is_fitted = 1;

    return self;
}

static Matrix* logreg_model_predict(const Estimator *self, const Matrix *X) {
    const LogisticRegressionModel *model = (const LogisticRegressionModel *)self;
    if (!model->base.is_fitted || !model->weights) {
        cml_set_error(CML_ERROR_NOT_FITTED, "logreg_model_predict: model not fitted");
        return NULL;
    }

    return logreg_predict(X, model->weights, 0.5);
}

static Matrix* logreg_model_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const LogisticRegressionModel *model = (const LogisticRegressionModel *)self;
    if (!model->base.is_fitted || !model->weights) {
        cml_set_error(CML_ERROR_NOT_FITTED, "logreg_model_predict_proba: model not fitted");
        return NULL;
    }

    /* Build a (n_samples x 2) probability matrix: [P(0), P(1)] */
    Matrix *p1 = logreg_predict_proba(X, model->weights);
    if (!p1) return NULL;

    size_t n = p1->rows;
    Matrix *proba = matrix_alloc(n, 2);
    if (!proba) {
        matrix_free(p1);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        double p = p1->data[i];
        proba->data[i * 2 + 0] = 1.0 - p;   /* P(class 0) */
        proba->data[i * 2 + 1] = p;          /* P(class 1) */
    }

    matrix_free(p1);
    return proba;
}

static double logreg_model_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

static Estimator* logreg_model_clone(const Estimator *self) {
    /* Return an unfitted clone with the same hyperparameters */
    (void)self;
    return (Estimator *)logreg_model_create();
}

static void logreg_model_free(Estimator *self) {
    LogisticRegressionModel *model = (LogisticRegressionModel *)self;
    if (model) {
        if (model->weights) {
            matrix_free(model->weights);
        }
        free(model);
    }
}

/* ============================================
 * LogisticRegression save/load
 * ============================================ */

int logreg_model_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const LogisticRegressionModel *model = (const LogisticRegressionModel *)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_LOGREG_BINARY) != 0) { fclose(f); return -1; }

    cml_ser_write_double(f, model->learning_rate);
    cml_ser_write_int(f, model->max_iter);
    cml_ser_write_double(f, model->lambda_val);
    cml_ser_write_double(f, model->tol);
    cml_ser_write_int(f, model->n_features);
    cml_ser_write_int(f, model->n_classes);
    cml_ser_write_matrix(f, model->weights);

    fclose(f);
    return 0;
}

Estimator* logreg_model_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_LOGREG_BINARY) != 0) { fclose(f); return NULL; }

    double lr, lambda_val, tol;
    int max_iter, n_features, n_classes;

    if (cml_ser_read_double(f, &lr) != 0)          { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &max_iter) != 0)        { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &lambda_val) != 0)   { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &tol) != 0)          { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)      { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_classes) != 0)       { fclose(f); return NULL; }

    LogisticRegressionModel *model = logreg_model_create();
    if (!model) { fclose(f); return NULL; }

    model->learning_rate = lr;
    model->max_iter      = max_iter;
    model->lambda_val    = lambda_val;
    model->tol           = tol;
    model->n_features    = n_features;
    model->n_classes     = n_classes;

    model->weights = cml_ser_read_matrix(f);
    if (!model->weights) { logreg_model_free((Estimator*)model); fclose(f); return NULL; }

    fclose(f);
    model->base.is_fitted = 1;
    return (Estimator*)model;
}

/* ============================================
 * Softmax (Multi-class) Logistic Regression
 * ============================================ */

static Estimator*  softmax_model_fit(Estimator *self, const Matrix *X, const Matrix *y);
static Matrix*     softmax_model_predict(const Estimator *self, const Matrix *X);
static Matrix*     softmax_model_predict_proba_impl(const Estimator *self, const Matrix *X);
static double      softmax_model_score(const Estimator *self, const Matrix *X, const Matrix *y);
static Estimator*  softmax_model_clone(const Estimator *self);
static void        softmax_model_free(Estimator *self);
int         softmax_model_save(const Estimator *self, const char *path);
Estimator*  softmax_model_load(const char *path);

SoftmaxRegressionModel* softmax_model_create(void) {
    SoftmaxRegressionModel *model = calloc(1, sizeof(SoftmaxRegressionModel));
    if (!model) return NULL;

    model->base.type        = MODEL_LOGISTIC_REGRESSION;
    model->base.task        = TASK_CLASSIFICATION;
    model->base.is_fitted   = 0;
    model->base.verbose     = VERBOSE_SILENT;

    model->base.fit            = softmax_model_fit;
    model->base.predict        = softmax_model_predict;
    model->base.predict_proba  = softmax_model_predict_proba_impl;
    model->base.transform      = NULL;
    model->base.score          = softmax_model_score;
    model->base.clone          = softmax_model_clone;
    model->base.free           = softmax_model_free;
    model->base.save           = softmax_model_save;
    model->base.load           = softmax_model_load;
    model->base.print_summary  = NULL;

    /* Default hyperparameters */
    model->learning_rate = 0.1;
    model->max_iter      = 1000;
    model->tol           = 1e-6;

    model->weights    = NULL;
    model->biases     = NULL;
    model->n_features = 0;
    model->n_classes  = 0;

    return model;
}

static void softmax_free_params(SoftmaxRegressionModel *model) {
    if (!model) return;
    if (model->weights) {
        matrix_free(model->weights);
        model->weights = NULL;
    }
    free(model->biases);
    model->biases = NULL;
    model->n_features = 0;
    model->n_classes = 0;
}

static Estimator* softmax_model_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    SoftmaxRegressionModel *model = (SoftmaxRegressionModel *)self;

    if (!X || !y || X->rows != y->rows) {
        return NULL;
    }

    /* Free previous state if re-fitting */
    softmax_free_params(model);

    size_t n = X->rows;
    size_t m = X->cols;

    /* Detect n_classes from unique values in y */
    int max_class = 0;
    for (size_t i = 0; i < n; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    int C = max_class + 1;

    /* Initialize weights (C × m) and biases (C) to zero */
    Matrix *W = matrix_alloc((size_t)C, m);
    if (!W) return NULL;

    double *b = calloc((size_t)C, sizeof(double));
    if (!b) {
        matrix_free(W);
        return NULL;
    }

    /* One-hot encode y → Y (n × C) */
    Matrix *Y = matrix_alloc(n, (size_t)C);
    if (!Y) {
        matrix_free(W);
        free(b);
        return NULL;
    }
    /* matrix_alloc zeros the data, so just set the 1s */
    for (size_t i = 0; i < n; i++) {
        int c = (int)y->data[i];
        Y->data[i * (size_t)C + c] = 1.0;
    }

    double lr = model->learning_rate;
    int max_iter = model->max_iter;
    double tol = model->tol;

    /* Wt = W^T (m × C) for computing Z = X * W^T */
    for (int iter = 0; iter < max_iter; iter++) {
        /* Compute Z = X * W^T + b  → (n × C)
         * We compute Z manually using W directly (C × m) */
        Matrix *Z = matrix_alloc(n, (size_t)C);
        if (!Z) {
            matrix_free(Y);
            matrix_free(W);
            free(b);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                double z_val = b[c];
                for (size_t j = 0; j < m; j++) {
                    z_val += X->data[i * m + j] * W->data[(size_t)c * m + j];
                }
                Z->data[i * (size_t)C + c] = z_val;
            }
        }

        /* Apply softmax with log-sum-exp trick */
        Matrix *P = matrix_alloc(n, (size_t)C);
        if (!P) {
            matrix_free(Z);
            matrix_free(Y);
            matrix_free(W);
            free(b);
            return NULL;
        }

        for (size_t i = 0; i < n; i++) {
            /* Find max per row for numerical stability */
            double max_z = Z->data[i * (size_t)C];
            for (int c = 1; c < C; c++) {
                double val = Z->data[i * (size_t)C + c];
                if (val > max_z) max_z = val;
            }

            double sum_exp = 0.0;
            for (int c = 0; c < C; c++) {
                double val = exp(Z->data[i * (size_t)C + c] - max_z);
                P->data[i * (size_t)C + c] = val;
                sum_exp += val;
            }
            for (int c = 0; c < C; c++) {
                P->data[i * (size_t)C + c] /= sum_exp;
            }
        }

        /* Compute diff = P - Y  (n × C) */
        Matrix *diff = matrix_alloc(n, (size_t)C);
        if (!diff) {
            matrix_free(P);
            matrix_free(Z);
            matrix_free(Y);
            matrix_free(W);
            free(b);
            return NULL;
        }

        for (size_t i = 0; i < n * (size_t)C; i++) {
            diff->data[i] = P->data[i] - Y->data[i];
        }

        /* Gradient: dW = diff^T * X / n  (C × n) × (n × m) = (C × m) */
        double max_grad = 0.0;
        for (int c = 0; c < C; c++) {
            double db = 0.0;
            for (size_t j = 0; j < m; j++) {
                double dw = 0.0;
                for (size_t i = 0; i < n; i++) {
                    dw += diff->data[i * (size_t)C + c] * X->data[i * m + j];
                }
                dw /= (double)n;
                double grad_abs = fabs(dw);
                if (grad_abs > max_grad) max_grad = grad_abs;
                W->data[(size_t)c * m + j] -= lr * dw;
            }
            /* db = mean(diff per class) */
            for (size_t i = 0; i < n; i++) {
                db += diff->data[i * (size_t)C + c];
            }
            db /= (double)n;
            {
                double grad_abs = fabs(db);
                if (grad_abs > max_grad) max_grad = grad_abs;
            }
            b[c] -= lr * db;
        }

        matrix_free(diff);
        matrix_free(P);
        matrix_free(Z);

        /* Check convergence */
        if (max_grad < tol) {
            break;
        }
    }

    matrix_free(Y);

    model->weights    = W;
    model->biases     = b;
    model->n_features = (int)m;
    model->n_classes  = C;
    model->base.is_fitted = 1;

    return self;
}

static Matrix* softmax_model_predict_proba_impl(const Estimator *self, const Matrix *X) {
    const SoftmaxRegressionModel *model = (const SoftmaxRegressionModel *)self;
    if (!model->base.is_fitted || !model->weights || !model->biases) return NULL;

    size_t n = X->rows;
    int C = model->n_classes;
    int m = model->n_features;

    Matrix *P = matrix_alloc(n, (size_t)C);
    if (!P) return NULL;

    for (size_t i = 0; i < n; i++) {
        /* Compute Z = X[i] * W^T + b */
        double max_z = -1e300;
        for (int c = 0; c < C; c++) {
            double z_val = model->biases[c];
            for (int j = 0; j < m; j++) {
                z_val += X->data[i * (size_t)m + j] * model->weights->data[(size_t)c * (size_t)m + j];
            }
            P->data[i * (size_t)C + c] = z_val;
            if (z_val > max_z) max_z = z_val;
        }

        double sum_exp = 0.0;
        for (int c = 0; c < C; c++) {
            double val = exp(P->data[i * (size_t)C + c] - max_z);
            P->data[i * (size_t)C + c] = val;
            sum_exp += val;
        }
        for (int c = 0; c < C; c++) {
            P->data[i * (size_t)C + c] /= sum_exp;
        }
    }

    return P;
}

static Matrix* softmax_model_predict(const Estimator *self, const Matrix *X) {
    Matrix *P = softmax_model_predict_proba_impl(self, X);
    if (!P) return NULL;

    size_t n = P->rows;
    size_t C = P->cols;

    Matrix *pred = matrix_alloc(n, 1);
    if (!pred) {
        matrix_free(P);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        int best_c = 0;
        double best_p = P->data[i * C];
        for (size_t c = 1; c < C; c++) {
            if (P->data[i * C + c] > best_p) {
                best_p = P->data[i * C + c];
                best_c = (int)c;
            }
        }
        pred->data[i] = (double)best_c;
    }

    matrix_free(P);
    return pred;
}

static double softmax_model_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

static Estimator* softmax_model_clone(const Estimator *self) {
    (void)self;
    return (Estimator *)softmax_model_create();
}

static void softmax_model_free(Estimator *self) {
    SoftmaxRegressionModel *model = (SoftmaxRegressionModel *)self;
    if (model) {
        softmax_free_params(model);
        free(model);
    }
}

/* ============================================
 * Softmax save/load
 * ============================================ */

int softmax_model_save(const Estimator *self, const char *path) {
    if (!self || !path || !self->is_fitted) return -1;
    const SoftmaxRegressionModel *model = (const SoftmaxRegressionModel *)self;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (cml_ser_write_header(f, CML_SUB_SOFTMAX) != 0) { fclose(f); return -1; }

    cml_ser_write_double(f, model->learning_rate);
    cml_ser_write_int(f, model->max_iter);
    cml_ser_write_double(f, model->tol);
    cml_ser_write_int(f, model->n_features);
    cml_ser_write_int(f, model->n_classes);
    cml_ser_write_matrix(f, model->weights);
    cml_ser_write_doubles(f, model->biases, model->n_classes);

    fclose(f);
    return 0;
}

Estimator* softmax_model_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (cml_ser_check_header(f, CML_SUB_SOFTMAX) != 0) { fclose(f); return NULL; }

    double lr, tol;
    int max_iter, n_features, n_classes;

    if (cml_ser_read_double(f, &lr) != 0)      { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &max_iter) != 0)    { fclose(f); return NULL; }
    if (cml_ser_read_double(f, &tol) != 0)     { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_features) != 0)  { fclose(f); return NULL; }
    if (cml_ser_read_int(f, &n_classes) != 0)   { fclose(f); return NULL; }

    SoftmaxRegressionModel *model = softmax_model_create();
    if (!model) { fclose(f); return NULL; }

    model->learning_rate = lr;
    model->max_iter      = max_iter;
    model->tol           = tol;
    model->n_features    = n_features;
    model->n_classes     = n_classes;

    model->weights = cml_ser_read_matrix(f);
    if (!model->weights) { softmax_model_free((Estimator*)model); fclose(f); return NULL; }

    model->biases = malloc((size_t)n_classes * sizeof(double));
    if (!model->biases) { softmax_model_free((Estimator*)model); fclose(f); return NULL; }
    if (cml_ser_read_doubles(f, model->biases, n_classes) != 0) {
        softmax_model_free((Estimator*)model); fclose(f); return NULL;
    }

    fclose(f);
    model->base.is_fitted = 1;
    return (Estimator*)model;
}
