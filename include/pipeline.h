/**
 * pipeline.h - Pipeline for chaining transformers and estimators
 *
 * Provides scikit-learn style Pipeline:
 * - Chain multiple preprocessing steps with a final estimator
 * - Single fit/predict interface
 * - Automatic data transformation through each step
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include "matrix.h"
#include "estimator.h"
#include "preprocessing.h"

/**
 * Pipeline step types
 */
typedef enum {
    STEP_TRANSFORMER,   // Only transform (e.g., scaler)
    STEP_ESTIMATOR      // Can fit and predict (e.g., model)
} PipelineStepType;

/**
 * Transformer interface (for preprocessing steps)
 */
typedef struct Transformer {
    const char *name;

    // Fit the transformer to data
    struct Transformer* (*fit)(struct Transformer *self, const Matrix *X);

    // Transform data
    Matrix* (*transform)(const struct Transformer *self, const Matrix *X);

    // Fit and transform in one step
    Matrix* (*fit_transform)(struct Transformer *self, const Matrix *X);

    // Inverse transform (if applicable)
    Matrix* (*inverse_transform)(const struct Transformer *self, const Matrix *X);

    // Clone transformer
    struct Transformer* (*clone)(const struct Transformer *self);

    // Free transformer
    void (*free)(struct Transformer *self);

    int is_fitted;
    void *data;  // Transformer-specific data
} Transformer;

/**
 * Pipeline step
 */
typedef struct {
    char *name;
    PipelineStepType type;
    union {
        Transformer *transformer;
        Estimator *estimator;
    } step;
} PipelineStep;

/**
 * Pipeline structure
 */
typedef struct {
    Estimator base;         // Inherit from Estimator
    PipelineStep *steps;
    int n_steps;
    int capacity;
} Pipeline;

/* ============================================
 * Standard Transformers (Preprocessing)
 * ============================================ */

/**
 * Standard Scaler - standardize features by removing mean and scaling to unit variance
 */
typedef struct {
    Transformer base;
    Scaler *scaler;
} StandardScalerTransformer;

StandardScalerTransformer* standard_scaler_transformer_create(void);

/**
 * MinMax Scaler - scale features to [0, 1] range
 */
typedef struct {
    Transformer base;
    MinMaxScaler *scaler;
} MinMaxScalerTransformer;

MinMaxScalerTransformer* minmax_scaler_transformer_create(void);

/**
 * Polynomial Features - generate polynomial and interaction features
 */
typedef struct {
    Transformer base;
    int degree;
    int include_bias;
    int interaction_only;
    size_t n_input_features;
    size_t n_output_features;
} PolynomialFeatures;

PolynomialFeatures* polynomial_features_create(int degree, int include_bias, int interaction_only);

/* ============================================
 * Pipeline API
 * ============================================ */

/**
 * Create a new pipeline
 */
Pipeline* pipeline_create(void);

/**
 * Add a transformer step to the pipeline
 */
int pipeline_add_transformer(Pipeline *pipe, const char *name, Transformer *transformer);

/**
 * Add an estimator as the final step
 */
int pipeline_add_estimator(Pipeline *pipe, const char *name, Estimator *estimator);

/**
 * Fit the pipeline (fit all transformers and the final estimator)
 */
Estimator* pipeline_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform data through all transformers
 */
Matrix* pipeline_transform(const Pipeline *pipe, const Matrix *X);

/**
 * Predict using the final estimator
 */
Matrix* pipeline_predict(const Estimator *self, const Matrix *X);

/**
 * Score using the final estimator
 */
double pipeline_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Fit and transform through the pipeline
 */
Matrix* pipeline_fit_transform(Pipeline *pipe, const Matrix *X, const Matrix *y);

/**
 * Clone the pipeline
 */
Estimator* pipeline_clone(const Estimator *self);

/**
 * Free the pipeline
 */
void pipeline_free(Estimator *self);

/**
 * Print pipeline summary
 */
void pipeline_print_summary(const Estimator *self);

/**
 * Get a step by name
 */
PipelineStep* pipeline_get_step(Pipeline *pipe, const char *name);

/**
 * Get named step as Transformer
 */
Transformer* pipeline_get_transformer(Pipeline *pipe, const char *name);

/**
 * Get the final estimator
 */
Estimator* pipeline_get_estimator(Pipeline *pipe);

/* ============================================
 * Convenience function: make_pipeline
 * ============================================ */

/**
 * Create a pipeline with standard scaler and linear regression
 *
 * Example usage:
 *   Pipeline *pipe = make_pipeline(
 *       "scaler", (void*)standard_scaler_transformer_create(),
 *       "model", (void*)linear_regression_create(LINREG_SOLVER_AUTO),
 *       NULL
 *   );
 */
// Note: Variadic function not easily done in C without tricks
// Instead, use pipeline_create() + pipeline_add_*()

#endif /* PIPELINE_H */
