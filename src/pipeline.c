/**
 * pipeline.c - Pipeline implementation
 */

#include "pipeline.h"
#include "cml_error.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Standard Scaler Transformer
 * ============================================ */

static Transformer* standard_scaler_fit(Transformer *self, const Matrix *X) {
    StandardScalerTransformer *sst = (StandardScalerTransformer*)self;

    if (sst->scaler) {
        scaler_free(sst->scaler);
    }
    sst->scaler = standardize_fit(X);
    sst->base.is_fitted = (sst->scaler != NULL);

    return self;
}

static Matrix* standard_scaler_transform(const Transformer *self, const Matrix *X) {
    const StandardScalerTransformer *sst = (const StandardScalerTransformer*)self;

    if (!sst->base.is_fitted || !sst->scaler) {
        cml_set_error(CML_ERROR_NOT_FITTED, "StandardScaler not fitted");
        return NULL;
    }

    return standardize_transform(X, sst->scaler);
}

static Matrix* standard_scaler_fit_transform(Transformer *self, const Matrix *X) {
    StandardScalerTransformer *sst = (StandardScalerTransformer*)self;

    if (sst->scaler) {
        scaler_free(sst->scaler);
    }

    Matrix *result = standardize_fit_transform(X, &sst->scaler);
    sst->base.is_fitted = (sst->scaler != NULL);

    return result;
}

static Transformer* standard_scaler_clone(const Transformer *self) {
    (void)self;
    return (Transformer*)standard_scaler_transformer_create();
}

static void standard_scaler_free(Transformer *self) {
    StandardScalerTransformer *sst = (StandardScalerTransformer*)self;
    if (sst) {
        scaler_free(sst->scaler);
        free(sst);
    }
}

StandardScalerTransformer* standard_scaler_transformer_create(void) {
    StandardScalerTransformer *sst = calloc(1, sizeof(StandardScalerTransformer));
    if (!sst) return NULL;

    sst->base.name = "StandardScaler";
    sst->base.fit = standard_scaler_fit;
    sst->base.transform = standard_scaler_transform;
    sst->base.fit_transform = standard_scaler_fit_transform;
    sst->base.inverse_transform = NULL;  // TODO: implement
    sst->base.clone = standard_scaler_clone;
    sst->base.free = standard_scaler_free;
    sst->base.is_fitted = 0;
    sst->scaler = NULL;

    return sst;
}

/* ============================================
 * MinMax Scaler Transformer
 * ============================================ */

static Transformer* minmax_scaler_fit(Transformer *self, const Matrix *X) {
    MinMaxScalerTransformer *mst = (MinMaxScalerTransformer*)self;

    if (mst->scaler) {
        minmax_scaler_free(mst->scaler);
    }
    mst->scaler = minmax_fit(X);
    mst->base.is_fitted = (mst->scaler != NULL);

    return self;
}

static Matrix* minmax_scaler_transform(const Transformer *self, const Matrix *X) {
    const MinMaxScalerTransformer *mst = (const MinMaxScalerTransformer*)self;

    if (!mst->base.is_fitted || !mst->scaler) {
        cml_set_error(CML_ERROR_NOT_FITTED, "MinMaxScaler not fitted");
        return NULL;
    }

    return minmax_transform(X, mst->scaler);
}

static Matrix* minmax_scaler_fit_transform(Transformer *self, const Matrix *X) {
    MinMaxScalerTransformer *mst = (MinMaxScalerTransformer*)self;

    mst->base.fit(self, X);
    return mst->base.transform(self, X);
}

static Transformer* minmax_scaler_clone(const Transformer *self) {
    (void)self;
    return (Transformer*)minmax_scaler_transformer_create();
}

static void minmax_scaler_free_impl(Transformer *self) {
    MinMaxScalerTransformer *mst = (MinMaxScalerTransformer*)self;
    if (mst) {
        minmax_scaler_free(mst->scaler);
        free(mst);
    }
}

MinMaxScalerTransformer* minmax_scaler_transformer_create(void) {
    MinMaxScalerTransformer *mst = calloc(1, sizeof(MinMaxScalerTransformer));
    if (!mst) return NULL;

    mst->base.name = "MinMaxScaler";
    mst->base.fit = minmax_scaler_fit;
    mst->base.transform = minmax_scaler_transform;
    mst->base.fit_transform = minmax_scaler_fit_transform;
    mst->base.inverse_transform = NULL;
    mst->base.clone = minmax_scaler_clone;
    mst->base.free = minmax_scaler_free_impl;
    mst->base.is_fitted = 0;
    mst->scaler = NULL;

    return mst;
}

/* ============================================
 * Polynomial Features Transformer
 * ============================================ */

static size_t count_polynomial_features(int n_features, int degree, int include_bias, int interaction_only) {
    // For degree=2 with n features: 1 (bias) + n (linear) + n*(n+1)/2 (quadratic)
    // This is a simplified count for degree=2
    size_t count = 0;

    if (include_bias) count++;

    // Linear terms
    count += n_features;

    // Quadratic terms (for degree >= 2)
    if (degree >= 2) {
        if (interaction_only) {
            // Only x_i * x_j where i < j
            count += (n_features * (n_features - 1)) / 2;
        } else {
            // x_i * x_j for all i <= j (includes x_i^2)
            count += (n_features * (n_features + 1)) / 2;
        }
    }

    return count;
}

static Transformer* polynomial_fit(Transformer *self, const Matrix *X) {
    PolynomialFeatures *pf = (PolynomialFeatures*)self;

    pf->n_input_features = X->cols;
    pf->n_output_features = count_polynomial_features(
        X->cols, pf->degree, pf->include_bias, pf->interaction_only
    );
    pf->base.is_fitted = 1;

    return self;
}

static Matrix* polynomial_transform(const Transformer *self, const Matrix *X) {
    const PolynomialFeatures *pf = (const PolynomialFeatures*)self;

    if (!pf->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "PolynomialFeatures not fitted");
        return NULL;
    }

    size_t n = X->rows;
    size_t m = pf->n_output_features;

    Matrix *result = matrix_alloc(n, m);
    if (!result) return NULL;

    for (size_t i = 0; i < n; i++) {
        size_t col_idx = 0;

        // Bias term
        if (pf->include_bias) {
            result->data[i * m + col_idx++] = 1.0;
        }

        // Linear terms
        for (size_t j = 0; j < X->cols; j++) {
            result->data[i * m + col_idx++] = X->data[i * X->cols + j];
        }

        // Quadratic terms (for degree >= 2)
        if (pf->degree >= 2) {
            for (size_t j = 0; j < X->cols; j++) {
                size_t start_k = pf->interaction_only ? j + 1 : j;
                for (size_t k = start_k; k < X->cols; k++) {
                    double val_j = X->data[i * X->cols + j];
                    double val_k = X->data[i * X->cols + k];
                    result->data[i * m + col_idx++] = val_j * val_k;
                }
            }
        }
    }

    return result;
}

static Matrix* polynomial_fit_transform(Transformer *self, const Matrix *X) {
    self->fit(self, X);
    return self->transform(self, X);
}

static Transformer* polynomial_clone(const Transformer *self) {
    const PolynomialFeatures *pf = (const PolynomialFeatures*)self;
    return (Transformer*)polynomial_features_create(pf->degree, pf->include_bias, pf->interaction_only);
}

static void polynomial_free(Transformer *self) {
    free(self);
}

PolynomialFeatures* polynomial_features_create(int degree, int include_bias, int interaction_only) {
    PolynomialFeatures *pf = calloc(1, sizeof(PolynomialFeatures));
    if (!pf) return NULL;

    pf->base.name = "PolynomialFeatures";
    pf->base.fit = polynomial_fit;
    pf->base.transform = polynomial_transform;
    pf->base.fit_transform = polynomial_fit_transform;
    pf->base.inverse_transform = NULL;
    pf->base.clone = polynomial_clone;
    pf->base.free = polynomial_free;
    pf->base.is_fitted = 0;

    pf->degree = degree;
    pf->include_bias = include_bias;
    pf->interaction_only = interaction_only;
    pf->n_input_features = 0;
    pf->n_output_features = 0;

    return pf;
}

/* ============================================
 * Pipeline Implementation
 * ============================================ */

Pipeline* pipeline_create(void) {
    Pipeline *pipe = calloc(1, sizeof(Pipeline));
    if (!pipe) return NULL;

    // Set up as Estimator
    pipe->base.type = MODEL_LINEAR_REGRESSION;  // Will be overridden by final estimator
    pipe->base.task = TASK_REGRESSION;
    pipe->base.is_fitted = 0;
    pipe->base.verbose = VERBOSE_SILENT;

    pipe->base.fit = pipeline_fit;
    pipe->base.predict = pipeline_predict;
    pipe->base.predict_proba = NULL;
    pipe->base.transform = NULL;
    pipe->base.score = pipeline_score;
    pipe->base.clone = pipeline_clone;
    pipe->base.free = pipeline_free;
    pipe->base.save = NULL;
    pipe->base.load = NULL;
    pipe->base.print_summary = pipeline_print_summary;

    pipe->steps = NULL;
    pipe->n_steps = 0;
    pipe->capacity = 0;

    return pipe;
}

int pipeline_add_transformer(Pipeline *pipe, const char *name, Transformer *transformer) {
    if (!pipe || !name || !transformer) return -1;

    // Expand capacity if needed
    if (pipe->n_steps >= pipe->capacity) {
        int new_capacity = pipe->capacity == 0 ? 4 : pipe->capacity * 2;
        PipelineStep *new_steps = realloc(pipe->steps, new_capacity * sizeof(PipelineStep));
        if (!new_steps) return -1;
        pipe->steps = new_steps;
        pipe->capacity = new_capacity;
    }

    PipelineStep *step = &pipe->steps[pipe->n_steps];
    step->name = cml_strdup(name);
    if (!step->name) { return -1; }
    step->type = STEP_TRANSFORMER;
    step->step.transformer = transformer;
    pipe->n_steps++;

    return 0;
}

int pipeline_add_estimator(Pipeline *pipe, const char *name, Estimator *estimator) {
    if (!pipe || !name || !estimator) return -1;

    // Check if last step is already an estimator
    if (pipe->n_steps > 0 && pipe->steps[pipe->n_steps - 1].type == STEP_ESTIMATOR) {
        cml_set_error(CML_ERROR_INVALID_ARG, "Pipeline already has a final estimator");
        return -1;
    }

    // Expand capacity if needed
    if (pipe->n_steps >= pipe->capacity) {
        int new_capacity = pipe->capacity == 0 ? 4 : pipe->capacity * 2;
        PipelineStep *new_steps = realloc(pipe->steps, new_capacity * sizeof(PipelineStep));
        if (!new_steps) return -1;
        pipe->steps = new_steps;
        pipe->capacity = new_capacity;
    }

    PipelineStep *step = &pipe->steps[pipe->n_steps];
    step->name = cml_strdup(name);
    if (!step->name) { return -1; }
    step->type = STEP_ESTIMATOR;
    step->step.estimator = estimator;
    pipe->n_steps++;

    // Update pipeline type from estimator
    pipe->base.type = estimator->type;
    pipe->base.task = estimator->task;

    return 0;
}

Estimator* pipeline_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    Pipeline *pipe = (Pipeline*)self;

    if (pipe->n_steps == 0) {
        cml_set_error(CML_ERROR_INVALID_ARG, "Pipeline is empty");
        return NULL;
    }

    Matrix *X_transformed = matrix_copy(X);
    if (!X_transformed) return NULL;

    // Fit and transform through all steps except the last
    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];

        if (step->type == STEP_TRANSFORMER) {
            Matrix *X_new = step->step.transformer->fit_transform(step->step.transformer, X_transformed);
            matrix_free(X_transformed);
            if (!X_new) return NULL;
            X_transformed = X_new;
        } else if (step->type == STEP_ESTIMATOR) {
            // Final estimator - just fit
            step->step.estimator->fit(step->step.estimator, X_transformed, y);
        }
    }

    matrix_free(X_transformed);
    pipe->base.is_fitted = 1;

    return self;
}

Matrix* pipeline_transform(const Pipeline *pipe, const Matrix *X) {
    Matrix *X_transformed = matrix_copy(X);
    if (!X_transformed) return NULL;

    // Transform through all transformer steps
    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];

        if (step->type == STEP_TRANSFORMER) {
            Matrix *X_new = step->step.transformer->transform(step->step.transformer, X_transformed);
            matrix_free(X_transformed);
            if (!X_new) return NULL;
            X_transformed = X_new;
        }
        // Stop at estimator (don't predict here)
        if (step->type == STEP_ESTIMATOR) break;
    }

    return X_transformed;
}

Matrix* pipeline_predict(const Estimator *self, const Matrix *X) {
    const Pipeline *pipe = (const Pipeline*)self;

    if (!pipe->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "Pipeline not fitted");
        return NULL;
    }

    // Transform through all steps
    Matrix *X_transformed = matrix_copy(X);
    if (!X_transformed) return NULL;

    Estimator *final_estimator = NULL;

    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];

        if (step->type == STEP_TRANSFORMER) {
            Matrix *X_new = step->step.transformer->transform(step->step.transformer, X_transformed);
            matrix_free(X_transformed);
            if (!X_new) return NULL;
            X_transformed = X_new;
        } else if (step->type == STEP_ESTIMATOR) {
            final_estimator = step->step.estimator;
        }
    }

    if (!final_estimator) {
        matrix_free(X_transformed);
        cml_set_error(CML_ERROR_INVALID_ARG, "Pipeline has no final estimator");
        return NULL;
    }

    Matrix *predictions = final_estimator->predict(final_estimator, X_transformed);
    matrix_free(X_transformed);

    return predictions;
}

double pipeline_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    const Pipeline *pipe = (const Pipeline*)self;

    if (!pipe->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "Pipeline not fitted");
        return -1.0;
    }

    // Transform and get final estimator
    Matrix *X_transformed = matrix_copy(X);
    if (!X_transformed) return -1.0;

    Estimator *final_estimator = NULL;

    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];

        if (step->type == STEP_TRANSFORMER) {
            Matrix *X_new = step->step.transformer->transform(step->step.transformer, X_transformed);
            matrix_free(X_transformed);
            if (!X_new) return -1.0;
            X_transformed = X_new;
        } else if (step->type == STEP_ESTIMATOR) {
            final_estimator = step->step.estimator;
        }
    }

    if (!final_estimator) {
        matrix_free(X_transformed);
        return -1.0;
    }

    double score = final_estimator->score(final_estimator, X_transformed, y);
    matrix_free(X_transformed);

    return score;
}

Estimator* pipeline_clone(const Estimator *self) {
    const Pipeline *pipe = (const Pipeline*)self;

    Pipeline *clone = pipeline_create();
    if (!clone) return NULL;

    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];

        if (step->type == STEP_TRANSFORMER) {
            Transformer *t_clone = step->step.transformer->clone(step->step.transformer);
            if (!t_clone || pipeline_add_transformer(clone, step->name, t_clone) != 0) {
                pipeline_free((Estimator*)clone);
                return NULL;
            }
        } else if (step->type == STEP_ESTIMATOR) {
            Estimator *e_clone = step->step.estimator->clone(step->step.estimator);
            if (!e_clone || pipeline_add_estimator(clone, step->name, e_clone) != 0) {
                pipeline_free((Estimator*)clone);
                return NULL;
            }
        }
    }

    return (Estimator*)clone;
}

void pipeline_free(Estimator *self) {
    Pipeline *pipe = (Pipeline*)self;
    if (!pipe) return;

    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];
        free(step->name);

        if (step->type == STEP_TRANSFORMER) {
            step->step.transformer->free(step->step.transformer);
        } else if (step->type == STEP_ESTIMATOR) {
            step->step.estimator->free(step->step.estimator);
        }
    }

    free(pipe->steps);
    free(pipe);
}

void pipeline_print_summary(const Estimator *self) {
    const Pipeline *pipe = (const Pipeline*)self;

    printf("\n=== Pipeline Summary ===\n");
    printf("Fitted: %s\n", pipe->base.is_fitted ? "Yes" : "No");
    printf("Steps: %d\n\n", pipe->n_steps);

    for (int i = 0; i < pipe->n_steps; i++) {
        PipelineStep *step = &pipe->steps[i];
        const char *type_str = step->type == STEP_TRANSFORMER ? "Transformer" : "Estimator";

        printf("  [%d] %s (%s)\n", i + 1, step->name, type_str);

        if (step->type == STEP_TRANSFORMER) {
            printf("      Type: %s\n", step->step.transformer->name);
            printf("      Fitted: %s\n", step->step.transformer->is_fitted ? "Yes" : "No");
        } else if (step->type == STEP_ESTIMATOR && step->step.estimator->print_summary) {
            // Don't print full summary, just type
            printf("      Fitted: %s\n", step->step.estimator->is_fitted ? "Yes" : "No");
        }
    }
    printf("========================\n\n");
}

PipelineStep* pipeline_get_step(Pipeline *pipe, const char *name) {
    if (!pipe || !name) return NULL;

    for (int i = 0; i < pipe->n_steps; i++) {
        if (strcmp(pipe->steps[i].name, name) == 0) {
            return &pipe->steps[i];
        }
    }
    return NULL;
}

Transformer* pipeline_get_transformer(Pipeline *pipe, const char *name) {
    PipelineStep *step = pipeline_get_step(pipe, name);
    if (step && step->type == STEP_TRANSFORMER) {
        return step->step.transformer;
    }
    return NULL;
}

Estimator* pipeline_get_estimator(Pipeline *pipe) {
    if (!pipe || pipe->n_steps == 0) return NULL;

    PipelineStep *last = &pipe->steps[pipe->n_steps - 1];
    if (last->type == STEP_ESTIMATOR) {
        return last->step.estimator;
    }
    return NULL;
}
