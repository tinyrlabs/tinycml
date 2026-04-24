/**
 * @file linear_regression.c
 * @brief Implementation of linear regression with Estimator API
 */

#include "linear_regression.h"
#include "preprocessing.h"
#include "metrics.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ============================================
 * Internal helper functions
 * ============================================ */

/* Matrix inversion using Gauss-Jordan elimination */
static Matrix* matrix_inverse(const Matrix *m) {
    if (!m || m->rows != m->cols) {
        return NULL;
    }

    size_t n = m->rows;
    Matrix *augmented = matrix_alloc(n, 2 * n);
    if (!augmented) {
        return NULL;
    }

    /* Create augmented matrix [A | I] */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented->data[i * 2 * n + j] = m->data[i * n + j];
        }
        augmented->data[i * 2 * n + n + i] = 1.0;
    }

    /* Gauss-Jordan elimination */
    for (size_t col = 0; col < n; col++) {
        /* Find pivot */
        size_t max_row = col;
        double max_val = fabs(augmented->data[col * 2 * n + col]);
        for (size_t row = col + 1; row < n; row++) {
            double val = fabs(augmented->data[row * 2 * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        /* Check for singularity */
        if (max_val < 1e-10) {
            matrix_free(augmented);
            return NULL;
        }

        /* Swap rows */
        if (max_row != col) {
            for (size_t j = 0; j < 2 * n; j++) {
                double temp = augmented->data[col * 2 * n + j];
                augmented->data[col * 2 * n + j] = augmented->data[max_row * 2 * n + j];
                augmented->data[max_row * 2 * n + j] = temp;
            }
        }

        /* Scale pivot row */
        double pivot = augmented->data[col * 2 * n + col];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented->data[col * 2 * n + j] /= pivot;
        }

        /* Eliminate column */
        for (size_t row = 0; row < n; row++) {
            if (row != col) {
                double factor = augmented->data[row * 2 * n + col];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented->data[row * 2 * n + j] -= factor * augmented->data[col * 2 * n + j];
                }
            }
        }
    }

    /* Extract inverse from right half */
    Matrix *inverse = matrix_alloc(n, n);
    if (!inverse) {
        matrix_free(augmented);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverse->data[i * n + j] = augmented->data[i * 2 * n + n + j];
        }
    }

    matrix_free(augmented);
    return inverse;
}

/* Compute MSE loss */
static double compute_mse_loss(const Matrix *X, const Matrix *y, const Matrix *weights) {
    Matrix *pred = matrix_matmul(X, weights);
    if (!pred) return INFINITY;

    double loss = 0.0;
    for (size_t i = 0; i < y->rows; i++) {
        double diff = pred->data[i] - y->data[i];
        loss += diff * diff;
    }
    matrix_free(pred);
    return loss / y->rows;
}

/* ============================================
 * Estimator API Implementation
 * ============================================ */

LinearRegression* linear_regression_create(LinRegSolver solver) {
    return linear_regression_create_full(solver, 0.01, 1000, 1e-4, 1, 0);
}

LinearRegression* linear_regression_create_full(
    LinRegSolver solver,
    double learning_rate,
    int max_iter,
    double tol,
    int fit_intercept,
    int normalize
) {
    LinearRegression *model = calloc(1, sizeof(LinearRegression));
    if (!model) return NULL;

    // Set up base estimator
    model->base.type = MODEL_LINEAR_REGRESSION;
    model->base.task = TASK_REGRESSION;
    model->base.is_fitted = 0;
    model->base.verbose = VERBOSE_SILENT;

    // Set function pointers (virtual function table)
    model->base.fit = linear_regression_fit;
    model->base.predict = linear_regression_predict;
    model->base.predict_proba = NULL;  // Not applicable
    model->base.transform = NULL;       // Not applicable
    model->base.score = linear_regression_score;
    model->base.clone = linear_regression_clone;
    model->base.free = linear_regression_free;
    model->base.save = linear_regression_save;
    model->base.load = NULL;  // Static function
    model->base.print_summary = linear_regression_print_summary;

    // Set hyperparameters
    model->solver = solver;
    model->learning_rate = learning_rate;
    model->max_iter = max_iter;
    model->tol = tol;
    model->fit_intercept = fit_intercept;
    model->normalize = normalize;

    // Initialize model data
    model->weights = NULL;
    model->coef_ = NULL;
    model->intercept_ = 0.0;
    model->n_iter_ = 0;

    return model;
}

Estimator* linear_regression_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return NULL;

    LinearRegression *model = (LinearRegression*)self;
    clock_t start = clock();

    // Prepare data - add bias if needed
    Matrix *X_fit = NULL;
    if (model->fit_intercept) {
        X_fit = add_bias_column(X);
    } else {
        X_fit = matrix_copy(X);
    }

    if (!X_fit) return NULL;

    // Free previous weights if refitting
    if (model->weights) {
        matrix_free(model->weights);
        model->weights = NULL;
    }
    if (model->coef_) {
        matrix_free(model->coef_);
        model->coef_ = NULL;
    }

    // Select solver
    LinRegSolver solver = model->solver;
    if (solver == LINREG_SOLVER_AUTO) {
        // Use closed form for small data, GD for large
        solver = (X->rows * X->cols < 10000) ? LINREG_SOLVER_CLOSED : LINREG_SOLVER_GD;
    }

    // Allocate history for iterative solvers
    if (solver != LINREG_SOLVER_CLOSED && model->base.verbose != VERBOSE_SILENT) {
        if (model->base.history) {
            training_history_free(model->base.history);
        }
        model->base.history = training_history_alloc(model->max_iter);
    }

    // Train based on solver
    if (solver == LINREG_SOLVER_CLOSED) {
        model->weights = linreg_fit_closed(X_fit, y);
        model->n_iter_ = 1;

        if (model->base.verbose >= VERBOSE_MINIMAL) {
            double loss = compute_mse_loss(X_fit, y, model->weights);
            printf("LinearRegression (closed-form) - MSE: %.6f\n", loss);
        }
    } else {
        // Gradient descent with convergence check
        size_t n = X_fit->rows;
        size_t m = X_fit->cols;

        model->weights = matrix_alloc(m, 1);
        if (!model->weights) {
            matrix_free(X_fit);
            return NULL;
        }

        Matrix *Xt = matrix_transpose(X_fit);
        if (!Xt) {
            matrix_free(X_fit);
            matrix_free(model->weights);
            model->weights = NULL;
            return NULL;
        }

        double prev_loss = INFINITY;

        for (int epoch = 0; epoch < model->max_iter; epoch++) {
            // Compute predictions
            Matrix *pred = matrix_matmul(X_fit, model->weights);
            if (!pred) break;

            // Compute error
            Matrix *error = matrix_sub(pred, y);
            matrix_free(pred);
            if (!error) break;

            // Compute gradient
            Matrix *gradient = matrix_matmul(Xt, error);
            matrix_free(error);
            if (!gradient) break;

            // Update weights
            for (size_t i = 0; i < m; i++) {
                model->weights->data[i] -= model->learning_rate * gradient->data[i] / (double)n;
            }
            matrix_free(gradient);

            // Compute loss for convergence check
            double loss = compute_mse_loss(X_fit, y, model->weights);
            double r2 = 1.0 - loss;  // Approximate R² for history

            // Record history
            if (model->base.history) {
                training_history_append(model->base.history, loss, r2);
            }

            // Callback
            if (model->base.callback) {
                model->base.callback(epoch + 1, loss, r2, model->base.callback_data);
            }

            // Verbose output
            verbose_print_epoch(model->base.verbose, epoch + 1, model->max_iter,
                              loss, r2, "R²");

            // Check convergence
            if (fabs(prev_loss - loss) < model->tol) {
                if (model->base.history) {
                    model->base.history->converged = 1;
                    model->base.history->best_epoch = epoch;
                }
                model->n_iter_ = epoch + 1;
                if (model->base.verbose >= VERBOSE_MINIMAL) {
                    printf("Converged at epoch %d\n", epoch + 1);
                }
                break;
            }
            prev_loss = loss;
            model->n_iter_ = epoch + 1;
        }

        matrix_free(Xt);
    }

    matrix_free(X_fit);

    if (!model->weights) {
        return NULL;
    }

    // Extract coefficients and intercept
    size_t n_coef = model->fit_intercept ? model->weights->rows - 1 : model->weights->rows;
    model->coef_ = matrix_alloc(n_coef, 1);
    if (model->coef_) {
        size_t offset = model->fit_intercept ? 1 : 0;
        for (size_t i = 0; i < n_coef; i++) {
            model->coef_->data[i] = model->weights->data[i + offset];
        }
    }

    if (model->fit_intercept) {
        model->intercept_ = model->weights->data[0];
    } else {
        model->intercept_ = 0.0;
    }

    model->base.is_fitted = 1;

    // Final verbose output
    if (model->base.verbose >= VERBOSE_MINIMAL) {
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double final_score = linear_regression_score(self, X, y);
        verbose_print_final(model->base.verbose, "LinearRegression", final_score, "R²", elapsed);
    }

    return self;
}

Matrix* linear_regression_predict(const Estimator *self, const Matrix *X) {
    if (!estimator_check_fitted(self, "predict")) return NULL;

    const LinearRegression *model = (const LinearRegression*)self;

    // Add bias column if model was trained with intercept
    Matrix *X_pred = NULL;
    if (model->fit_intercept) {
        X_pred = add_bias_column(X);
    } else {
        X_pred = matrix_copy(X);
    }

    if (!X_pred) return NULL;

    Matrix *predictions = matrix_matmul(X_pred, model->weights);
    matrix_free(X_pred);

    return predictions;
}

double linear_regression_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* linear_regression_clone(const Estimator *self) {
    if (!self) return NULL;
    const LinearRegression *model = (const LinearRegression*)self;

    return (Estimator*)linear_regression_create_full(
        model->solver,
        model->learning_rate,
        model->max_iter,
        model->tol,
        model->fit_intercept,
        model->normalize
    );
}

void linear_regression_free(Estimator *self) {
    if (!self) return;
    LinearRegression *model = (LinearRegression*)self;

    matrix_free(model->weights);
    matrix_free(model->coef_);
    training_history_free(model->base.history);
    free(model);
}

int linear_regression_save(const Estimator *self, const char *filename) {
    if (!self || !filename) return -1;
    if (!self->is_fitted) return -1;

    const LinearRegression *model = (const LinearRegression*)self;
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    // Write magic number and version
    const char magic[] = "TCML";
    fwrite(magic, 1, 4, f);
    int version = 1;
    fwrite(&version, sizeof(int), 1, f);

    // Write model type
    fwrite(&model->base.type, sizeof(ModelType), 1, f);

    // Write hyperparameters
    fwrite(&model->solver, sizeof(LinRegSolver), 1, f);
    fwrite(&model->learning_rate, sizeof(double), 1, f);
    fwrite(&model->max_iter, sizeof(int), 1, f);
    fwrite(&model->tol, sizeof(double), 1, f);
    fwrite(&model->fit_intercept, sizeof(int), 1, f);
    fwrite(&model->normalize, sizeof(int), 1, f);

    // Write weights
    fwrite(&model->weights->rows, sizeof(size_t), 1, f);
    fwrite(&model->weights->cols, sizeof(size_t), 1, f);
    fwrite(model->weights->data, sizeof(double), model->weights->rows * model->weights->cols, f);

    fclose(f);
    return 0;
}

Estimator* linear_regression_load(const char *filename) {
    if (!filename) return NULL;

    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    // Verify magic number
    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strcmp(magic, "TCML") != 0) {
        fclose(f);
        return NULL;
    }

    // Read version
    int version;
    if (fread(&version, sizeof(int), 1, f) != 1 || version != 1) {
        fclose(f);
        return NULL;
    }

    // Read model type
    ModelType type;
    if (fread(&type, sizeof(ModelType), 1, f) != 1 || type != MODEL_LINEAR_REGRESSION) {
        fclose(f);
        return NULL;
    }

    // Read hyperparameters
    LinRegSolver solver;
    double learning_rate, tol;
    int max_iter, fit_intercept, normalize;

    if (fread(&solver, sizeof(LinRegSolver), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&learning_rate, sizeof(double), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&max_iter, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&tol, sizeof(double), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&fit_intercept, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&normalize, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    // Create model
    LinearRegression *model = linear_regression_create_full(
        solver, learning_rate, max_iter, tol, fit_intercept, normalize
    );
    if (!model) {
        fclose(f);
        return NULL;
    }

    // Read weights
    size_t rows, cols;
    if (fread(&rows, sizeof(size_t), 1, f) != 1) { linear_regression_free((Estimator*)model); fclose(f); return NULL; }
    if (fread(&cols, sizeof(size_t), 1, f) != 1) { linear_regression_free((Estimator*)model); fclose(f); return NULL; }

    model->weights = matrix_alloc(rows, cols);
    if (!model->weights) {
        linear_regression_free((Estimator*)model);
        fclose(f);
        return NULL;
    }
    if (fread(model->weights->data, sizeof(double), rows * cols, f) != rows * cols) { linear_regression_free((Estimator*)model); fclose(f); return NULL; }

    fclose(f);

    // Extract coefficients
    size_t n_coef = fit_intercept ? rows - 1 : rows;
    model->coef_ = matrix_alloc(n_coef, 1);
    if (model->coef_) {
        size_t offset = fit_intercept ? 1 : 0;
        for (size_t i = 0; i < n_coef; i++) {
            model->coef_->data[i] = model->weights->data[i + offset];
        }
    }
    model->intercept_ = fit_intercept ? model->weights->data[0] : 0.0;
    model->base.is_fitted = 1;

    return (Estimator*)model;
}

void linear_regression_print_summary(const Estimator *self) {
    if (!self) return;
    const LinearRegression *model = (const LinearRegression*)self;

    printf("\n=== LinearRegression Summary ===\n");
    printf("Fitted: %s\n", model->base.is_fitted ? "Yes" : "No");

    const char *solver_names[] = {"closed-form", "gradient descent", "SGD", "auto"};
    printf("Solver: %s\n", solver_names[model->solver]);

    if (model->solver != LINREG_SOLVER_CLOSED) {
        printf("Learning rate: %.6f\n", model->learning_rate);
        printf("Max iterations: %d\n", model->max_iter);
        printf("Tolerance: %.2e\n", model->tol);
    }

    printf("Fit intercept: %s\n", model->fit_intercept ? "Yes" : "No");

    if (model->base.is_fitted) {
        printf("\nIntercept: %.6f\n", model->intercept_);
        printf("Coefficients (%zu):\n", model->coef_->rows);
        for (size_t i = 0; i < model->coef_->rows && i < 10; i++) {
            printf("  [%zu]: %.6f\n", i, model->coef_->data[i]);
        }
        if (model->coef_->rows > 10) {
            printf("  ... (%zu more)\n", model->coef_->rows - 10);
        }
        printf("Iterations: %d\n", model->n_iter_);
    }
    printf("================================\n\n");
}

const Matrix* linear_regression_get_coef(const LinearRegression *model) {
    return model ? model->coef_ : NULL;
}

double linear_regression_get_intercept(const LinearRegression *model) {
    return model ? model->intercept_ : 0.0;
}

/* ============================================
 * Legacy API (backward compatible)
 * ============================================ */

Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y) {
    if (!X || !y || X->rows != y->rows) {
        return NULL;
    }

    Matrix *Xt = matrix_transpose(X);
    if (!Xt) return NULL;

    Matrix *XtX = matrix_matmul(Xt, X);
    if (!XtX) {
        matrix_free(Xt);
        return NULL;
    }

    Matrix *XtX_inv = matrix_inverse(XtX);
    if (!XtX_inv) {
        matrix_free(Xt);
        matrix_free(XtX);
        return NULL;
    }

    Matrix *Xty = matrix_matmul(Xt, y);
    if (!Xty) {
        matrix_free(Xt);
        matrix_free(XtX);
        matrix_free(XtX_inv);
        return NULL;
    }

    Matrix *weights = matrix_matmul(XtX_inv, Xty);

    matrix_free(Xt);
    matrix_free(XtX);
    matrix_free(XtX_inv);
    matrix_free(Xty);

    return weights;
}

Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs) {
    if (!X || !y || X->rows != y->rows || lr <= 0 || epochs <= 0) {
        return NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;

    Matrix *weights = matrix_alloc(m, 1);
    if (!weights) return NULL;

    Matrix *Xt = matrix_transpose(X);
    if (!Xt) {
        matrix_free(weights);
        return NULL;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix *pred = matrix_matmul(X, weights);
        if (!pred) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        Matrix *error = matrix_sub(pred, y);
        matrix_free(pred);
        if (!error) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        Matrix *gradient = matrix_matmul(Xt, error);
        matrix_free(error);
        if (!gradient) {
            matrix_free(weights);
            matrix_free(Xt);
            return NULL;
        }

        for (size_t i = 0; i < m; i++) {
            weights->data[i] -= lr * gradient->data[i] / (double)n;
        }

        matrix_free(gradient);
    }

    matrix_free(Xt);
    return weights;
}

Matrix* linreg_predict(const Matrix *X, const Matrix *weights) {
    if (!X || !weights || X->cols != weights->rows) {
        return NULL;
    }
    return matrix_matmul(X, weights);
}
