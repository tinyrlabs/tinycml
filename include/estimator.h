/**
 * estimator.h - Unified Estimator API for tinycml
 *
 * Provides a scikit-learn style interface for all models:
 * - fit(X, y) - Train the model
 * - predict(X) - Make predictions
 * - score(X, y) - Evaluate model performance
 *
 * Zero dependencies, pure C11.
 */

#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "matrix.h"

/**
 * Model type enumeration
 */
typedef enum {
    MODEL_LINEAR_REGRESSION,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_KNN,
    MODEL_KMEANS,
    MODEL_NAIVE_BAYES,
    MODEL_DECISION_TREE,
    MODEL_RANDOM_FOREST,
    MODEL_NEURAL_NETWORK,
    MODEL_SVM,
    MODEL_PCA,
    MODEL_FEATURE_SELECTOR
} ModelType;

/**
 * Task type - determines default scoring metric
 */
typedef enum {
    TASK_REGRESSION,
    TASK_CLASSIFICATION,
    TASK_CLUSTERING,
    TASK_TRANSFORMATION
} TaskType;

/**
 * Training history for iterative algorithms
 */
typedef struct {
    double *loss_history;      // Loss at each epoch
    double *metric_history;    // Primary metric at each epoch
    size_t n_epochs;           // Number of epochs recorded
    size_t capacity;           // Allocated capacity
    int converged;             // Whether training converged early
    size_t best_epoch;         // Epoch with best validation score
} TrainingHistory;

/**
 * Callback function type for training progress
 * Called at the end of each epoch with current state
 */
typedef void (*TrainingCallback)(
    int epoch,
    double loss,
    double metric,
    void *user_data
);

/**
 * Verbose level for training output
 */
typedef enum {
    VERBOSE_SILENT = 0,    // No output
    VERBOSE_MINIMAL = 1,   // Only final result
    VERBOSE_PROGRESS = 2,  // Progress bar / periodic updates
    VERBOSE_DETAILED = 3   // Every epoch
} VerboseLevel;

/**
 * Base estimator structure - all models embed this
 */
typedef struct Estimator {
    ModelType type;
    TaskType task;
    int is_fitted;

    // Virtual function table (polymorphism in C)
    struct Estimator* (*fit)(struct Estimator *self, const Matrix *X, const Matrix *y);
    Matrix* (*predict)(const struct Estimator *self, const Matrix *X);
    Matrix* (*predict_proba)(const struct Estimator *self, const Matrix *X);
    Matrix* (*transform)(const struct Estimator *self, const Matrix *X);
    double (*score)(const struct Estimator *self, const Matrix *X, const Matrix *y);
    struct Estimator* (*clone)(const struct Estimator *self);
    void (*free)(struct Estimator *self);
    int (*save)(const struct Estimator *self, const char *filename);
    struct Estimator* (*load)(const char *filename);
    void (*print_summary)(const struct Estimator *self);

    // Training configuration
    VerboseLevel verbose;
    TrainingCallback callback;
    void *callback_data;
    TrainingHistory *history;

    // Model-specific data follows this struct
    void *model_data;
} Estimator;

/**
 * Training history management
 */
TrainingHistory* training_history_alloc(size_t initial_capacity);
void training_history_append(TrainingHistory *history, double loss, double metric);
void training_history_free(TrainingHistory *history);
void training_history_print(const TrainingHistory *history);
int training_history_save_csv(const TrainingHistory *history, const char *filename);

/**
 * Estimator utilities
 */

// Check if model is fitted, return error message if not
int estimator_check_fitted(const Estimator *est, const char *method_name);

// Set verbose level
void estimator_set_verbose(Estimator *est, VerboseLevel level);

// Set training callback
void estimator_set_callback(Estimator *est, TrainingCallback cb, void *user_data);

// Get training history
const TrainingHistory* estimator_get_history(const Estimator *est);

// Clone estimator with same parameters but unfitted
Estimator* estimator_clone(const Estimator *est);

// Default score implementations
double regression_score_r2(const Estimator *est, const Matrix *X, const Matrix *y);
double classification_score_accuracy(const Estimator *est, const Matrix *X, const Matrix *y);

/**
 * Verbose output helpers
 */
void verbose_print_epoch(VerboseLevel level, int epoch, int total_epochs,
                         double loss, double metric, const char *metric_name);
void verbose_print_final(VerboseLevel level, const char *model_name,
                         double final_metric, const char *metric_name, double elapsed_time);
void verbose_print_progress_bar(int current, int total, int bar_width);

#endif // ESTIMATOR_H
