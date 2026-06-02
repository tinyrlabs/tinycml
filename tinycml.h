/* tinycml v0.1.0 - single-header ML library */
/* https://github.com/sametyilmaztemel/tinycml */
/*
 * SINGLE-HEADER AMALGAMATION
 *
 * Usage:
 *   #define CML_IMPLEMENTATION
 *   #include "tinycml.h"
 *
 * In exactly ONE translation unit, define CML_IMPLEMENTATION
 * before including this header to get the implementation.
 *
 * Without CML_IMPLEMENTATION, you only get the declarations.
 */

#ifndef TINYCML_H
#define TINYCML_H

/* === HEADERS === */

/* --- matrix.h --- */
/**
 * @file matrix.h
 * @brief Matrix data structure and operations for ML library
 *
 * Provides core matrix functionality including allocation, arithmetic,
 * and transformations. Uses row-major storage order.
 */


#include <stddef.h>

/**
 * @brief Matrix structure with row-major storage
 */
typedef struct {
    size_t rows;    /**< Number of rows */
    size_t cols;    /**< Number of columns */
    double *data;   /**< Row-major data array */
} Matrix;

/* Memory Management */

/**
 * @brief Allocate a zero-initialized matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to allocated matrix, or NULL on failure
 */
Matrix* matrix_alloc(size_t rows, size_t cols);

/**
 * @brief Free matrix memory
 * @param m Matrix to free
 */
void matrix_free(Matrix *m);

/**
 * @brief Create a deep copy of a matrix
 * @param m Matrix to copy
 * @return Pointer to new matrix, or NULL on failure
 */
Matrix* matrix_copy(const Matrix *m);

/* Element Access */

/**
 * @brief Get element at position (i, j)
 * @param m Matrix
 * @param i Row index
 * @param j Column index
 * @return Element value
 */
double matrix_get(const Matrix *m, size_t i, size_t j);

/**
 * @brief Set element at position (i, j)
 * @param m Matrix
 * @param i Row index
 * @param j Column index
 * @param val Value to set
 */
void matrix_set(Matrix *m, size_t i, size_t j, double val);

/* Arithmetic Operations */

/**
 * @brief Element-wise addition: C = A + B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_add(const Matrix *a, const Matrix *b);

/**
 * @brief Element-wise subtraction: C = A - B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_sub(const Matrix *a, const Matrix *b);

/**
 * @brief Element-wise multiplication (Hadamard product): C = A .* B
 * @param a First matrix
 * @param b Second matrix
 * @return Result matrix, or NULL on dimension mismatch
 */
Matrix* matrix_mul(const Matrix *a, const Matrix *b);

/**
 * @brief Scalar multiplication: C = scalar * A
 * @param m Matrix
 * @param scalar Scalar value
 * @return Result matrix, or NULL on failure
 */
Matrix* matrix_scale(const Matrix *m, double scalar);

/**
 * @brief Matrix multiplication: C = A * B
 * @param a First matrix (m x k)
 * @param b Second matrix (k x n)
 * @return Result matrix (m x n), or NULL on dimension mismatch
 */
Matrix* matrix_matmul(const Matrix *a, const Matrix *b);

/* Transformations */

/**
 * @brief Transpose matrix
 * @param m Matrix to transpose
 * @return Transposed matrix, or NULL on failure
 */
Matrix* matrix_transpose(const Matrix *m);

/* Utilities */

/**
 * @brief Print matrix to stdout
 * @param m Matrix to print
 */
void matrix_print(const Matrix *m);

/**
 * @brief Fill matrix with a constant value
 * @param m Matrix to fill
 * @param val Value to fill with
 */
void matrix_fill(Matrix *m, double val);

/**
 * @brief Create an identity matrix
 * @param n Size of the square matrix
 * @return n x n identity matrix, or NULL on failure
 */
Matrix* matrix_identity(size_t n);


/* --- utils.h --- */
/**
 * @file utils.h
 * @brief Utility functions for ML library
 *
 * Provides random number generation and statistical functions.
 */


#include <stddef.h>

/**
 * @brief Seed the random number generator
 * @param seed Random seed
 */
void rand_seed(unsigned int seed);

/**
 * @brief Generate uniform random number in [0, 1)
 * @return Random value
 */
double rand_uniform(void);

/**
 * @brief Generate uniform random number in [min, max)
 * @param min Minimum value
 * @param max Maximum value
 * @return Random value
 */
double rand_uniform_range(double min, double max);

/**
 * @brief Generate random number from standard normal distribution
 * @return Random value from N(0, 1)
 */
double rand_normal(void);

/**
 * @brief Generate random number from normal distribution
 * @param mean Mean of distribution
 * @param std Standard deviation
 * @return Random value from N(mean, std^2)
 */
double rand_normal_params(double mean, double std);

/**
 * @brief Compute mean of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Mean value
 */
double mean(const double *data, size_t n);

/**
 * @brief Compute standard deviation of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Standard deviation
 */
double std_dev(const double *data, size_t n);

/**
 * @brief Compute variance of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Variance
 */
double variance(const double *data, size_t n);

/**
 * @brief Fisher-Yates shuffle for an array of indices
 * @param indices Array to shuffle
 * @param n Number of elements
 */
void shuffle_indices(size_t *indices, size_t n);

/**
 * @brief Portable strdup replacement for C11 compatibility
 * @param s String to duplicate
 * @return Newly allocated copy of s, or NULL on failure
 */
char* cml_strdup(const char *s);


/* --- cml_error.h --- */
/**
 * @file cml_error.h
 * @brief Unified error handling for tinycml library
 *
 * Provides a thread-local error state with status codes, messages,
 * and convenience macros for common error patterns.
 */


#include <stdio.h>
#include <string.h>

/**
 * @brief Status codes for library operations
 */
typedef enum {
    CML_OK = 0,
    CML_ERROR_NULL_PTR,
    CML_ERROR_OUT_OF_BOUNDS,
    CML_ERROR_INVALID_ARG,
    CML_ERROR_MEMORY,
    CML_ERROR_DIMENSION_MISMATCH,
    CML_ERROR_NOT_FITTED,
    CML_ERROR_FILE_IO,
    CML_ERROR_PARSE,
    CML_ERROR_CONVERGENCE,
    CML_ERROR_UNSUPPORTED
} CMLStatus;

/**
 * @brief Stores the last error code + message
 */
typedef struct {
    CMLStatus code;
    char message[256];
} CMLError;

/**
 * @brief Get the last error
 * @return Copy of the last error struct
 */
CMLError cml_get_last_error(void);

/**
 * @brief Set the last error (variadic printf-style message)
 * @param code Status code
 * @param fmt printf-style format string
 */
void cml_set_error(CMLStatus code, const char *fmt, ...);

/**
 * @brief Get a human-readable string for a status code
 * @param code Status code
 * @return Static string describing the code
 */
const char* cml_status_string(CMLStatus code);

/* Convenience macros */

#define CML_CHECK_NULL(ptr, msg) do { \
    if (!(ptr)) { cml_set_error(CML_ERROR_NULL_PTR, "%s: NULL pointer", (msg)); return NULL; } \
} while(0)

#define CML_CHECK_FITTED(model, msg) do { \
    if (!(model)->fitted) { cml_set_error(CML_ERROR_NOT_FITTED, "%s: model not fitted", (msg)); return NULL; } \
} while(0)

#define CML_RETURN_ERROR(code, msg) do { \
    cml_set_error((code), "%s", (msg)); \
    return NULL; \
} while(0)


/* --- vector.h --- */
/**
 * @file vector.h
 * @brief Vector operations for ML library
 *
 * Provides common vector operations using Matrix (n x 1) representation.
 */



/**
 * @brief Compute dot product of two vectors
 * @param a First vector (n x 1 matrix)
 * @param b Second vector (n x 1 matrix)
 * @return Dot product value
 */
double vector_dot(const Matrix *a, const Matrix *b);

/**
 * @brief Compute L2 norm (Euclidean length) of a vector
 * @param v Vector (n x 1 matrix)
 * @return L2 norm
 */
double vector_norm(const Matrix *v);

/**
 * @brief Scale a vector by a scalar
 * @param v Vector (n x 1 matrix)
 * @param scalar Scalar value
 * @return Scaled vector, or NULL on failure
 */
Matrix* vector_scale(const Matrix *v, double scalar);

/**
 * @brief Add two vectors
 * @param a First vector
 * @param b Second vector
 * @return Sum vector, or NULL on failure
 */
Matrix* vector_add(const Matrix *a, const Matrix *b);

/**
 * @brief Subtract two vectors
 * @param a First vector
 * @param b Second vector
 * @return Difference vector, or NULL on failure
 */
Matrix* vector_sub(const Matrix *a, const Matrix *b);


/* --- csv.h --- */
/**
 * @file csv.h
 * @brief CSV file loading and saving for ML library
 */



/**
 * @brief Load a CSV file into a Matrix
 * @param filename Path to CSV file
 * @param has_header 1 if file has header row to skip, 0 otherwise
 * @return Matrix with loaded data, or NULL on error
 */
Matrix* csv_load(const char *filename, int has_header);

/**
 * @brief Save a Matrix to a CSV file
 * @param m Matrix to save
 * @param filename Output file path
 * @return 0 on success, -1 on error
 */
int csv_save(const Matrix *m, const char *filename);

/**
 * @brief Options for robust CSV loading
 */
typedef struct {
    int has_header;       /**< first row = header names */
    char delimiter;       /**< ',', '\t', ';', etc. 0 = auto-detect */
    char quote;           /**< '"' or '\0' for no quoting */
    double missing_val;   /**< value for empty/invalid fields (NAN) */
    int skip_empty_lines; /**< skip blank lines */
    int max_rows;         /**< 0 = unlimited */
    int max_cols;         /**< 0 = unlimited */
} CSVOptions;

/**
 * @brief Load CSV with options. Returns Matrix (rows x cols).
 * @param path Path to CSV file
 * @param opts Pointer to CSVOptions (NULL for defaults)
 * @return Matrix with loaded data, or NULL on error
 */
Matrix* csv_load_ext(const char *path, const CSVOptions *opts);

/**
 * @brief Get header names (valid after csv_load_ext with has_header=1)
 * @return Array of header name strings
 */
const char** csv_get_headers(void);

/**
 * @brief Get number of headers
 * @return Number of header columns
 */
int csv_get_header_count(void);

/**
 * @brief Auto-detect delimiter by examining first line
 * @param path Path to CSV file
 * @return Detected delimiter character
 */
char csv_detect_delimiter(const char *path);


/* --- metrics.h --- */
/**
 * @file metrics.h
 * @brief Evaluation metrics for ML models
 */



/**
 * @brief Confusion matrix structure
 */
typedef struct {
    int tp;  /**< True Positives */
    int tn;  /**< True Negatives */
    int fp;  /**< False Positives */
    int fn;  /**< False Negatives */
} ConfusionMatrix;

/* Regression Metrics */

/**
 * @brief Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MSE value
 */
double mse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Root Mean Squared Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return RMSE value
 */
double rmse(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Mean Absolute Error
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MAE value
 */
double mae(const Matrix *y_true, const Matrix *y_pred);

/* Classification Metrics */

/**
 * @brief Classification accuracy
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @return Accuracy (0 to 1)
 */
double accuracy(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Precision (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Precision value
 */
double precision(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Recall (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Recall value
 */
double recall(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief F1 Score (for binary classification)
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return F1 score
 */
double f1_score(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Compute confusion matrix
 * @param y_true True labels (0/1)
 * @param y_pred Predicted labels (0/1)
 * @return Confusion matrix
 */
ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Print confusion matrix
 * @param cm Confusion matrix to print
 */
void confusion_matrix_print(const ConfusionMatrix *cm);

/* Multi-class metrics */

/**
 * @brief Multi-class confusion matrix: returns n_classes x n_classes matrix
 * @param y_true True labels (integer class labels)
 * @param y_pred Predicted labels (integer class labels)
 * @return Dynamically allocated n_classes x n_classes Matrix (caller must free), or NULL on error
 *
 * Labels are assumed to be integers 0..n_classes-1. The matrix is auto-sized
 * to (max_label+1) x (max_label+1). Entry [i][j] = count of samples with
 * true class i predicted as class j.
 */
Matrix* confusion_matrix_multi(const Matrix *y_true, const Matrix *y_pred);

/**
 * @brief Macro-averaged precision across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged precision
 */
double precision_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Macro-averaged recall across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged recall
 */
double recall_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Macro-averaged F1 score across classes
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Macro-averaged F1 score
 */
double f1_score_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/**
 * @brief Weighted-averaged F1 score (weighted by class support)
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @param n_classes Number of classes
 * @return Weighted F1 score
 */
double f1_score_weighted(const Matrix *y_true, const Matrix *y_pred, int n_classes);

/* Clustering Metrics */

/**
 * @brief Compute mean Silhouette Coefficient for clustering
 * @param X Data matrix (n_samples x n_features)
 * @param labels Cluster labels (n_samples x 1), integer labels 0..k-1
 * @param n_clusters Number of clusters
 * @return Mean silhouette score in [-1, 1], or 0.0 on error
 *
 * For each sample i:
 *   a(i) = mean distance to other samples in same cluster
 *   b(i) = min over other clusters of mean distance to samples in that cluster
 *   s(i) = (b(i) - a(i)) / max(a(i), b(i))
 * Returns mean of s(i) over all samples.
 * Uses Euclidean distance.
 */
double silhouette_score(const Matrix *X, const Matrix *labels, int n_clusters);


/* --- kmeans.h --- */
/**
 * @file kmeans.h
 * @brief k-Means clustering
 */



/**
 * @brief k-Means model structure
 */
typedef struct {
    Matrix *centroids;  /**< Cluster centroids (k x n_features) */
    int k;              /**< Number of clusters */
    int max_iter;       /**< Maximum iterations */
} KMeansModel;

/**
 * @brief Fit k-Means model using Lloyd's algorithm
 * @param X Data matrix (n_samples x n_features)
 * @param k Number of clusters
 * @param max_iter Maximum iterations
 * @param seed Random seed for initialization
 * @return Fitted model, or NULL on error
 */
KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed);

/**
 * @brief Predict cluster labels for samples
 * @param model Fitted k-Means model
 * @param X Data matrix
 * @return Cluster labels (n_samples x 1), or NULL on error
 */
Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X);

/**
 * @brief Free k-Means model
 * @param model Model to free
 */
void kmeans_free(KMeansModel *model);


/* --- dbscan.h --- */
/**
 * @file dbscan.h
 * @brief DBSCAN density-based clustering
 */



/**
 * @brief DBSCAN clustering model
 */
typedef struct {
    double epsilon;     /**< Neighborhood radius (default 0.5) */
    int min_samples;    /**< Minimum neighbors (default 5) */
} DBSCAN;

/**
 * @brief Fit DBSCAN model to data
 * @param model DBSCAN model with epsilon and min_samples set
 * @param X Data matrix (n_samples x n_features)
 * @return Array of cluster labels (n_samples). -1 = noise. Caller must free.
 */
int* dbscan_fit(const DBSCAN *model, const Matrix *X);

/**
 * @brief Count number of distinct clusters in labels
 * @param labels Cluster label array
 * @param n Number of labels
 * @return Number of clusters (ignoring noise label -1)
 */
int dbscan_count_clusters(const int *labels, int n);

/**
 * @brief Create a DBSCAN model with default parameters
 * @return Newly allocated DBSCAN model, or NULL on failure
 */
DBSCAN* dbscan_create(void);

/**
 * @brief Free a DBSCAN model
 * @param model Model to free
 */
void dbscan_free(DBSCAN *model);


/* --- preprocessing.h --- */
/**
 * @file preprocessing.h
 * @brief Data preprocessing functions for ML library
 */



/**
 * @brief Result of train/test split
 */
typedef struct {
    Matrix *X_train;
    Matrix *X_test;
    Matrix *y_train;
    Matrix *y_test;
} TrainTestSplit;

/**
 * @brief Standard scaler (Z-score normalization)
 */
typedef struct {
    double *means;
    double *stds;
    size_t n_features;
} Scaler;

/**
 * @brief Min-max scaler
 */
typedef struct {
    double *mins;
    double *maxs;
    size_t n_features;
} MinMaxScaler;

/**
 * @brief Split data into training and test sets
 * @param X Feature matrix (n_samples x n_features)
 * @param y Target vector (n_samples x 1)
 * @param test_ratio Fraction for test set (e.g., 0.2)
 * @param seed Random seed for reproducibility
 * @return TrainTestSplit structure
 */
TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed);

/**
 * @brief Free memory in TrainTestSplit structure
 * @param split Split to free
 */
void train_test_split_free(TrainTestSplit *split);

/**
 * @brief Fit standard scaler to data
 * @param X Feature matrix
 * @return Fitted scaler, or NULL on error
 */
Scaler* standardize_fit(const Matrix *X);

/**
 * @brief Transform data using fitted scaler
 * @param X Feature matrix
 * @param scaler Fitted scaler
 * @return Transformed matrix, or NULL on error
 */
Matrix* standardize_transform(const Matrix *X, const Scaler *scaler);

/**
 * @brief Fit and transform in one step
 * @param X Feature matrix
 * @param scaler_out Output pointer for fitted scaler
 * @return Transformed matrix, or NULL on error
 */
Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out);

/**
 * @brief Free scaler memory
 * @param scaler Scaler to free
 */
void scaler_free(Scaler *scaler);

/**
 * @brief Fit min-max scaler to data
 * @param X Feature matrix
 * @return Fitted scaler, or NULL on error
 */
MinMaxScaler* minmax_fit(const Matrix *X);

/**
 * @brief Transform data using fitted min-max scaler
 * @param X Feature matrix
 * @param scaler Fitted scaler
 * @return Transformed matrix (scaled to [0,1]), or NULL on error
 */
Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler);

/**
 * @brief Free min-max scaler memory
 * @param scaler Scaler to free
 */
void minmax_scaler_free(MinMaxScaler *scaler);

/**
 * @brief Add bias column (ones) to matrix
 * @param X Feature matrix (n x m)
 * @return New matrix (n x m+1) with first column as ones
 */
Matrix* add_bias_column(const Matrix *X);


/* --- knn.h --- */
/**
 * @file knn.h
 * @brief k-Nearest Neighbors classifier
 */



/**
 * @brief k-NN model structure
 */
typedef struct {
    Matrix *X_train;  /**< Training features */
    Matrix *y_train;  /**< Training labels */
    int k;            /**< Number of neighbors */
} KNNModel;

/**
 * @brief Fit k-NN model (stores training data)
 * @param X Training feature matrix
 * @param y Training labels (class indices as doubles)
 * @param k Number of neighbors
 * @return Fitted model, or NULL on error
 */
KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k);

/**
 * @brief Predict class labels for new samples
 * @param model Fitted k-NN model
 * @param X Feature matrix to predict
 * @return Predicted class labels, or NULL on error
 */
Matrix* knn_predict(const KNNModel *model, const Matrix *X);

/**
 * @brief Predict class probabilities for new samples
 * @param model Fitted k-NN model
 * @param X Feature matrix to predict
 * @param n_classes Number of classes
 * @return n_samples x n_classes probability matrix, or NULL on error
 */
Matrix* knn_predict_proba(const KNNModel *model, const Matrix *X, int n_classes);

/**
 * @brief Free k-NN model
 * @param model Model to free
 */
void knn_free(KNNModel *model);


/* --- onehot_encoder.h --- */
/**
 * @file onehot_encoder.h
 * @brief One-hot encoding transformer for categorical integer features
 */



/**
 * @brief One-hot encoder for integer-valued categorical features
 *
 * Fits on a Matrix X where each column represents a categorical feature
 * with integer values. Transforms into a one-hot representation.
 */
typedef struct {
    int n_features;          /**< Number of input features (columns) */
    int *n_categories;       /**< Number of unique categories per feature */
    int **categories_;       /**< Category values per feature (sorted) */
    int total_output_cols;   /**< Sum of n_categories across all features */

    int is_fitted;           /**< Whether encoder has been fitted */
} OneHotEncoder;

/**
 * @brief Create a new OneHotEncoder
 * @return Pointer to newly allocated encoder, or NULL on failure
 */
OneHotEncoder* onehot_encoder_create(void);

/**
 * @brief Fit the encoder on training data
 * @param encoder Encoder to fit
 * @param X Feature matrix with integer-valued entries
 * @return 0 on success, -1 on error
 */
int onehot_encoder_fit(OneHotEncoder *encoder, const Matrix *X);

/**
 * @brief Transform data using fitted encoder
 * @param encoder Fitted encoder
 * @param X Feature matrix with integer-valued entries
 * @return One-hot encoded matrix (n_samples × total_output_cols), or NULL on error
 */
Matrix* onehot_encoder_transform(const OneHotEncoder *encoder, const Matrix *X);

/**
 * @brief Free encoder memory
 * @param encoder Encoder to free
 */
void onehot_encoder_free(OneHotEncoder *encoder);


/* --- estimator.h --- */
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


/* --- cml_serialization.h --- */
/**
 * cml_serialization.h - Shared serialization helpers for tinycml models
 *
 * Provides common binary format utilities for save/load:
 *   MAGIC (4 bytes: "CML\0")
 *   SUBTYPE (4 bytes: int)
 *   Model-specific data
 */


#include <stdio.h>

/* Magic bytes: "CML\0" */
#define CML_SER_MAGIC_SIZE 4

/* Model subtype IDs for distinguishing models sharing a ModelType */
enum {
    CML_SUB_LOGREG_BINARY     = 1,
    CML_SUB_SOFTMAX           = 2,
    CML_SUB_GAUSSIAN_NB       = 3,
    CML_SUB_MULTINOMIAL_NB    = 4,
    CML_SUB_LINEAR_SVC        = 5,
    CML_SUB_SVM_CLASSIFIER    = 6,
    CML_SUB_RIDGE             = 7,
    CML_SUB_LASSO             = 8,
    CML_SUB_DECISION_TREE_CLF = 9,
};

/** Write serialization header (magic + subtype). Returns 0 on success. */
int cml_ser_write_header(FILE *f, int subtype);

/** Read and verify serialization header. Returns 0 on success. */
int cml_ser_check_header(FILE *f, int expected_subtype);

/** Write a matrix in serialization format (rows, cols, data). */
int cml_ser_write_matrix(FILE *f, const Matrix *m);

/** Read a matrix from serialization format. Returns new matrix or NULL. */
Matrix* cml_ser_read_matrix(FILE *f);

/** Write an array of n doubles. */
int cml_ser_write_doubles(FILE *f, const double *data, int n);

/** Read an array of n doubles. */
int cml_ser_read_doubles(FILE *f, double *data, int n);

/** Write an array of n ints. */
int cml_ser_write_ints(FILE *f, const int *data, int n);

/** Read an array of n ints. */
int cml_ser_read_ints(FILE *f, int *data, int n);

/** Write a single double. */
int cml_ser_write_double(FILE *f, double val);

/** Read a single double. */
int cml_ser_read_double(FILE *f, double *val);

/** Write a single int. */
int cml_ser_write_int(FILE *f, int val);

/** Read a single int. */
int cml_ser_read_int(FILE *f, int *val);


/* --- linear_regression.h --- */
/**
 * @file linear_regression.h
 * @brief Linear regression for tinycml - Estimator API compatible
 */



/**
 * Solver method for linear regression
 */
typedef enum {
    LINREG_SOLVER_CLOSED,    // Normal equation (X'X)^(-1)X'y
    LINREG_SOLVER_GD,        // Gradient descent
    LINREG_SOLVER_SGD,       // Stochastic gradient descent
    LINREG_SOLVER_AUTO       // Auto-select based on data size
} LinRegSolver;

/**
 * Linear Regression model structure
 */
typedef struct {
    Estimator base;           // Inherit from Estimator

    // Model parameters
    Matrix *weights;          // Learned weights (m x 1)
    Matrix *coef_;            // Coefficients without intercept (m-1 x 1)
    double intercept_;        // Intercept term

    // Hyperparameters
    LinRegSolver solver;
    double learning_rate;
    int max_iter;
    double tol;               // Tolerance for convergence
    int fit_intercept;        // Whether to fit intercept
    int normalize;            // Whether to normalize features

    // Training info
    int n_iter_;              // Actual iterations used
} LinearRegression;

/**
 * Create a new Linear Regression model
 *
 * @param solver Solver method (CLOSED, GD, SGD, AUTO)
 * @return New LinearRegression model, or NULL on error
 */
LinearRegression* linear_regression_create(LinRegSolver solver);

/**
 * Create with full configuration
 */
LinearRegression* linear_regression_create_full(
    LinRegSolver solver,
    double learning_rate,
    int max_iter,
    double tol,
    int fit_intercept,
    int normalize
);

/**
 * Fit the model (Estimator API)
 * Automatically adds bias column if fit_intercept is true
 */
Estimator* linear_regression_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict target values (Estimator API)
 */
Matrix* linear_regression_predict(const Estimator *self, const Matrix *X);

/**
 * Score using R² (Estimator API)
 */
double linear_regression_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone the model (Estimator API)
 */
Estimator* linear_regression_clone(const Estimator *self);

/**
 * Free the model (Estimator API)
 */
void linear_regression_free(Estimator *self);

/**
 * Save model to file
 */
int linear_regression_save(const Estimator *self, const char *filename);

/**
 * Load model from file
 */
Estimator* linear_regression_load(const char *filename);

/**
 * Print model summary
 */
void linear_regression_print_summary(const Estimator *self);

/**
 * Get coefficients (without intercept)
 */
const Matrix* linear_regression_get_coef(const LinearRegression *model);

/**
 * Get intercept
 */
double linear_regression_get_intercept(const LinearRegression *model);

/* ============================================
 * Legacy API (backward compatible)
 * ============================================ */

/**
 * @brief Fit linear regression using closed-form solution (normal equation)
 * @deprecated Use linear_regression_create() + fit() instead
 */
Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y);

/**
 * @brief Fit linear regression using gradient descent
 * @deprecated Use linear_regression_create_full() + fit() instead
 */
Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict target values
 * @deprecated Use linear_regression_predict() instead
 */
Matrix* linreg_predict(const Matrix *X, const Matrix *weights);


/* --- logistic_regression.h --- */
/**
 * @file logistic_regression.h
 * @brief Logistic regression for binary classification
 */



/**
 * @brief Sigmoid function
 * @param x Input value
 * @return 1 / (1 + exp(-x))
 */
double sigmoid(double x);

/* ============================================
 * Legacy API (backward compatible)
 * ============================================ */

/**
 * @brief Fit logistic regression using gradient descent
 * @param X Feature matrix (n x m), should include bias column
 * @param y Binary target vector (n x 1), values 0 or 1
 * @param lr Learning rate
 * @param epochs Number of iterations
 * @return Weight vector (m x 1), or NULL on error
 */
Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs);

/**
 * @brief Predict probabilities
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @return Probability vector (n x 1), or NULL on error
 */
Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights);

/**
 * @brief Predict class labels
 * @param X Feature matrix (n x m)
 * @param weights Weight vector (m x 1)
 * @param threshold Classification threshold (usually 0.5)
 * @return Class labels (n x 1), values 0 or 1, or NULL on error
 */
Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold);

/* ============================================
 * Estimator-compatible API
 * ============================================ */

/**
 * @brief Logistic Regression model with Estimator vtable
 *
 * Embeds Estimator base so it can be used with Pipeline,
 * GridSearchCV, cross_val_score, etc.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double learning_rate;   /**< Learning rate for gradient descent (default 0.1) */
    int max_iter;           /**< Maximum number of iterations (default 1000) */
    double lambda_val;      /**< L2 regularization strength (default 0.0) */
    double tol;             /**< Convergence tolerance (default 1e-6) */

    /* Fitted parameters (set after fit) */
    Matrix *weights;        /**< Learned weight vector (n_features x 1) */
    int n_features;         /**< Number of features */
    int n_classes;          /**< Number of classes (2 for binary) */
} LogisticRegressionModel;

/**
 * @brief Create a new LogisticRegressionModel with default hyperparameters
 * @return Pointer to newly allocated model, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
LogisticRegressionModel* logreg_model_create(void);

/**
 * @brief Save fitted LogisticRegressionModel to binary file
 * @return 0 on success, -1 on error
 */
int logreg_model_save(const Estimator *self, const char *path);

/**
 * @brief Load LogisticRegressionModel from binary file
 * @return New fitted model, or NULL on error
 */
Estimator* logreg_model_load(const char *path);

/* ============================================
 * Softmax (Multi-class) Logistic Regression
 * ============================================ */

/**
 * @brief Softmax Regression model for multi-class classification
 *
 * Uses the softmax function to generalize logistic regression to multiple classes.
 * Embeds Estimator base so it can be used with Pipeline, cross_val_score, etc.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double learning_rate;   /**< Learning rate for gradient descent (default 0.1) */
    int max_iter;           /**< Maximum number of iterations (default 1000) */
    double tol;             /**< Convergence tolerance (default 1e-6) */

    /* Fitted parameters (set after fit) */
    Matrix *weights;        /**< Learned weight matrix (n_classes × n_features) */
    double *biases;         /**< Bias terms per class (n_classes) */
    int n_features;         /**< Number of features */
    int n_classes;          /**< Number of classes */
} SoftmaxRegressionModel;

/**
 * @brief Create a new SoftmaxRegressionModel with default hyperparameters
 * @return Pointer to newly allocated model, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
SoftmaxRegressionModel* softmax_model_create(void);

/**
 * @brief Save fitted SoftmaxRegressionModel to binary file
 * @return 0 on success, -1 on error
 */
int softmax_model_save(const Estimator *self, const char *path);

/**
 * @brief Load SoftmaxRegressionModel from binary file
 * @return New fitted model, or NULL on error
 */
Estimator* softmax_model_load(const char *path);


/* --- ridge.h --- */
/**
 * @file ridge.h
 * @brief Ridge (L2-regularized) regression for tinycml
 */



typedef struct {
    Estimator base;
    double alpha;      /* L2 penalty (default 1.0) */
    int max_iter;      /* unused (closed-form), kept for API compat */
    double tol;        /* unused (closed-form), kept for API compat */
    Matrix *weights;   /* fitted weights (n_features x 1) */
    double bias;       /* fitted intercept */
    int n_features;    /* number of features seen in fit */
} RidgeModel;

/**
 * @brief Create a new Ridge model with default alpha=1.0
 */
RidgeModel* ridge_model_create(void);

/**
 * @brief Estimator vtable: fit
 */
Estimator* ridge_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: predict
 */
Matrix* ridge_predict(const Estimator *self, const Matrix *X);

/**
 * @brief Estimator vtable: score (R²)
 */
double ridge_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: clone
 */
Estimator* ridge_clone(const Estimator *self);

/**
 * @brief Estimator vtable: free
 */
void ridge_free(Estimator *self);


/* --- lasso.h --- */
/**
 * @file lasso.h
 * @brief Lasso (L1-regularized) regression for tinycml
 */



typedef struct {
    Estimator base;
    double alpha;      /* L1 penalty (default 1.0) */
    int max_iter;      /* max coordinate-descent iterations */
    double tol;        /* convergence tolerance */
    Matrix *weights;   /* fitted weights (n_features x 1) */
    double bias;       /* fitted intercept */
    int n_features;    /* number of features seen in fit */
} LassoModel;

/**
 * @brief Create a new Lasso model with default alpha=1.0, max_iter=1000, tol=1e-6
 */
LassoModel* lasso_model_create(void);

/**
 * @brief Estimator vtable: fit
 */
Estimator* lasso_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: predict
 */
Matrix* lasso_predict(const Estimator *self, const Matrix *X);

/**
 * @brief Estimator vtable: score (R²)
 */
double lasso_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * @brief Estimator vtable: clone
 */
Estimator* lasso_clone(const Estimator *self);

/**
 * @brief Estimator vtable: free
 */
void lasso_free(Estimator *self);


/* --- svm.h --- */
/**
 * @file svm.h
 * @brief Linear Support Vector Machine classifier
 */



/**
 * @brief Linear SVM classifier using sub-gradient descent with hinge loss
 *
 * Supports binary classification with labels {-1, +1}.
 * If input labels are {0, 1}, they are converted internally.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double C;              /**< Regularization parameter (default 1.0) */
    double learning_rate;  /**< Learning rate for gradient descent (default 0.001) */
    int max_iter;          /**< Maximum number of epochs (default 1000) */
    double tol;            /**< Convergence tolerance (default 1e-4) */

    /* Fitted parameters */
    Matrix *weights;       /**< Weight vector (n_features x 1) */
    double bias;           /**< Bias term */
    int n_features;        /**< Number of features */

    /* Internal */
    int labels_converted;  /**< Whether input labels were converted from {0,1} to {-1,+1} */
} LinearSVC;

/**
 * @brief Create a new Linear SVM classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 */
LinearSVC* linear_svc_create(void);

/**
 * @brief Free a Linear SVM classifier
 * @param svc Classifier to free (passed as Estimator* for vtable compatibility)
 */
void linear_svc_free_impl(Estimator *self);

/**
 * @brief Save a fitted LinearSVC to a binary file
 * @param self Fitted Estimator pointer
 * @param path File path
 * @return 0 on success, -1 on error
 */
int linear_svc_save(const Estimator *self, const char *path);

/**
 * @brief Load a LinearSVC from a binary file
 * @param path File path
 * @return New fitted Estimator, or NULL on error
 */
Estimator* linear_svc_load(const char *path);

/**
 * @brief Predict class probabilities for new samples
 * @param model Fitted LinearSVC model
 * @param X Feature matrix
 * @return n_samples x 1 matrix of probabilities P(y=1), or NULL on error
 */
Matrix* linear_svc_predict_proba(const LinearSVC *model, const Matrix *X);

/* ============================================
 * SVM with kernel support (Linear + RBF)
 * ============================================ */

typedef enum {
    CML_KERNEL_LINEAR = 0,
    CML_KERNEL_RBF = 1
} SVMKernelType;

typedef struct {
    Estimator base;
    SVMKernelType kernel;
    double C;           /**< Regularization (default 1.0) */
    double gamma;       /**< RBF gamma (default 1.0/n_features, -1 = auto) */
    double lr;          /**< Learning rate for SGD */
    int max_iter;
    double tol;

    /* Fitted (linear kernel) */
    Matrix *weights;
    double bias;

    /* Fitted (kernel) */
    Matrix *support_vectors;  /**< n_support x n_features */
    double *alphas;           /**< Dual coefficients (stored with y_i baked in) */
    double *labels_train;     /**< Training labels y_i in {-1, +1} */
    int n_support;
    int n_features;

    int labels_converted;     /**< Whether input labels were converted */
} SVMClassifier;

/**
 * @brief Create a new SVM classifier with specified kernel
 * @param kernel Kernel type (CML_KERNEL_LINEAR or CML_KERNEL_RBF)
 * @return Pointer to newly allocated classifier, or NULL on failure
 */
SVMClassifier* svm_classifier_create(SVMKernelType kernel);

/**
 * @brief Free an SVMClassifier
 * @param self Estimator pointer to free
 */
void svm_classifier_free_impl(Estimator *self);

/**
 * @brief Save a fitted SVMClassifier to a binary file
 * @param self Fitted Estimator pointer
 * @param path File path
 * @return 0 on success, -1 on error
 */
int svm_classifier_save(const Estimator *self, const char *path);

/**
 * @brief Load an SVMClassifier from a binary file
 * @param path File path
 * @return New fitted Estimator, or NULL on error
 */
Estimator* svm_classifier_load(const char *path);


/* --- naive_bayes.h --- */
/**
 * @file naive_bayes.h
 * @brief Gaussian Naive Bayes classifier
 */



/**
 * @brief Gaussian Naive Bayes classifier
 *
 * Assumes features are conditionally independent given the class
 * and follow a Gaussian distribution.
 */
typedef struct {
    Estimator base;

    /* Training parameters */
    double var_smoothing;   /**< Variance smoothing added to variances (default 1e-9) */

    /* Fitted parameters (set after fit) */
    int n_classes;          /**< Number of unique classes */
    int n_features;         /**< Number of features */
    double *class_priors;       /**< P(class), size n_classes */
    int *class_count_;          /**< Sample count per class */
    double *class_log_prior_;   /**< log(P(class)) */
    double *theta_;             /**< Mean per (class, feature), n_classes × n_features row-major */
    double *var_;               /**< Variance per (class, feature), n_classes × n_features row-major */

    int total_samples_;         /**< Total number of training samples */
} GaussianNaiveBayes;

/**
 * @brief Create a new Gaussian Naive Bayes classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
GaussianNaiveBayes* gaussian_nb_create(void);

/**
 * @brief Save fitted GaussianNaiveBayes to binary file
 */
int gaussian_nb_save(const Estimator *self, const char *path);

/**
 * @brief Load GaussianNaiveBayes from binary file
 */
Estimator* gaussian_nb_load(const char *path);

/**
 * @brief Compute class-conditional probabilities for each sample
 * @param self Estimator pointer (must be fitted)
 * @param X Feature matrix (n_samples × n_features)
 * @return Matrix of probabilities (n_samples × n_classes), or NULL on error
 */
Matrix* gaussian_nb_predict_proba(const Estimator *self, const Matrix *X);

/* ============================================
 * Multinomial Naive Bayes
 * ============================================ */

/**
 * @brief Multinomial Naive Bayes classifier
 *
 * Suitable for count-based features (e.g., word counts in text classification).
 * Uses Laplace smoothing to avoid zero probabilities.
 */
typedef struct {
    Estimator base;

    /* Hyperparameters */
    double alpha;           /**< Laplace smoothing parameter (default 1.0) */

    /* Fitted parameters (set after fit) */
    Matrix *log_prior;      /**< Log prior probabilities (n_classes × 1) */
    Matrix *theta;          /**< Log-likelihood per (class, feature) (n_classes × n_features) */
    int n_classes;          /**< Number of unique classes */
    int n_features;         /**< Number of features */
} MultinomialNB;

/**
 * @brief Create a new MultinomialNB classifier with default parameters
 * @return Pointer to newly allocated classifier, or NULL on failure
 *
 * Uses Estimator interface via base field:
 *   base.fit, base.predict, base.score, base.predict_proba, base.clone, base.free
 */
MultinomialNB* multinomial_nb_create(void);

/**
 * @brief Save fitted MultinomialNB to binary file
 */
int multinomial_nb_save(const Estimator *self, const char *path);

/**
 * @brief Load MultinomialNB from binary file
 */
Estimator* multinomial_nb_load(const char *path);


/* --- decision_tree.h --- */
/**
 * decision_tree.h - Decision Tree for classification and regression
 *
 * Implements CART (Classification and Regression Trees):
 * - Binary splits using Gini impurity or MSE
 * - Recursive tree building
 * - Pruning support
 */



/**
 * Criterion for splitting
 */
typedef enum {
    CRITERION_GINI,     // Gini impurity (classification)
    CRITERION_ENTROPY,  // Information gain (classification)
    CRITERION_MSE,      // Mean squared error (regression)
    CRITERION_MAE       // Mean absolute error (regression)
} SplitCriterion;

/**
 * Tree node structure
 */
typedef struct TreeNode {
    int is_leaf;

    // Split info (for internal nodes)
    size_t feature_index;
    double threshold;
    struct TreeNode *left;
    struct TreeNode *right;

    // Prediction info (for leaf nodes)
    double value;           // For regression: mean value
    int class_label;        // For classification: majority class
    double *class_proba;    // Class probabilities
    int n_classes;

    // Statistics
    size_t n_samples;
    double impurity;
} TreeNode;

/**
 * Decision Tree Classifier
 */
typedef struct {
    Estimator base;

    // Tree structure
    TreeNode *root;

    // Hyperparameters
    SplitCriterion criterion;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    double min_impurity_decrease;
    int max_features;       // 0 = all features

    // Training info
    int n_classes;
    int n_features;
    int depth_;
    int n_nodes_;
    int n_leaves_;
} DecisionTreeClassifier;

/**
 * Decision Tree Regressor
 */
typedef struct {
    Estimator base;

    TreeNode *root;

    SplitCriterion criterion;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    double min_impurity_decrease;
    int max_features;

    int n_features;
    int depth_;
    int n_nodes_;
    int n_leaves_;
} DecisionTreeRegressor;

/* ============================================
 * Decision Tree Classifier API
 * ============================================ */

/**
 * Create a Decision Tree Classifier
 */
DecisionTreeClassifier* decision_tree_classifier_create(void);

/**
 * Create with full configuration
 */
DecisionTreeClassifier* decision_tree_classifier_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
);

/**
 * Fit the classifier
 */
Estimator* decision_tree_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict class labels
 */
Matrix* decision_tree_classifier_predict(const Estimator *self, const Matrix *X);

/**
 * Predict class probabilities
 */
Matrix* decision_tree_classifier_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score (accuracy)
 */
double decision_tree_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* decision_tree_classifier_clone(const Estimator *self);

/**
 * Free
 */
void decision_tree_classifier_free(Estimator *self);

/**
 * Print tree structure
 */
void decision_tree_classifier_print_tree(const DecisionTreeClassifier *tree);

/**
 * Get feature importances
 */
Matrix* decision_tree_get_feature_importances(const DecisionTreeClassifier *tree);

/* ============================================
 * Decision Tree Regressor API
 * ============================================ */

/**
 * Create a Decision Tree Regressor
 */
DecisionTreeRegressor* decision_tree_regressor_create(void);

/**
 * Create with full configuration
 */
DecisionTreeRegressor* decision_tree_regressor_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
);

/**
 * Fit the regressor
 */
Estimator* decision_tree_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict values
 */
Matrix* decision_tree_regressor_predict(const Estimator *self, const Matrix *X);

/**
 * Score (R²)
 */
double decision_tree_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* decision_tree_regressor_clone(const Estimator *self);

/**
 * Free
 */
void decision_tree_regressor_free(Estimator *self);

/* ============================================
 * Tree utilities
 * ============================================ */

/**
 * Free a tree node recursively
 */
void tree_node_free(TreeNode *node);

/**
 * Export tree to text format
 */
void decision_tree_export_text(const TreeNode *root, int depth);


/* --- decomposition.h --- */
/**
 * decomposition.h - Dimensionality reduction algorithms
 *
 * Provides:
 * - PCA (Principal Component Analysis)
 * - Truncated SVD (for sparse data)
 */



/**
 * PCA (Principal Component Analysis)
 *
 * Linear dimensionality reduction using Singular Value Decomposition
 */
typedef struct {
    Estimator base;

    // Configuration
    int n_components;           // Number of components to keep
    int whiten;                 // Whether to whiten components

    // Fitted attributes
    Matrix *components_;        // Principal axes (n_components x n_features)
    Matrix *mean_;              // Per-feature mean (1 x n_features)
    double *explained_variance_;         // Variance explained by each component
    double *explained_variance_ratio_;   // Percentage of variance explained
    double *singular_values_;            // Singular values
    int n_features_;
    int n_samples_;
    double total_variance_;
} PCA;

/* ============================================
 * PCA API
 * ============================================ */

/**
 * Create PCA transformer
 *
 * @param n_components Number of components to keep (0 = keep all)
 * @return PCA instance
 */
PCA* pca_create(int n_components);

/**
 * Create with whitening
 */
PCA* pca_create_full(int n_components, int whiten);

/**
 * Fit PCA to data
 */
Estimator* pca_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform data to reduced dimensionality
 */
Matrix* pca_transform(const Estimator *self, const Matrix *X);

/**
 * Fit and transform in one step
 */
Matrix* pca_fit_transform(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Inverse transform - reconstruct original data from reduced
 */
Matrix* pca_inverse_transform(const PCA *pca, const Matrix *X_reduced);

/**
 * Get explained variance ratio
 */
const double* pca_explained_variance_ratio(const PCA *pca);

/**
 * Get cumulative explained variance
 */
double pca_cumulative_variance(const PCA *pca, int n_components);

/**
 * Clone PCA
 */
Estimator* pca_clone(const Estimator *self);

/**
 * Free PCA
 */
void pca_free(Estimator *self);

/**
 * Print PCA summary
 */
void pca_print_summary(const Estimator *self);

/* ============================================
 * Helper functions
 * ============================================ */

/**
 * Compute covariance matrix
 */
Matrix* compute_covariance_matrix(const Matrix *X, const Matrix *mean);

/**
 * Power iteration for dominant eigenvector
 */
Matrix* power_iteration(const Matrix *A, int max_iter, double tol);

/**
 * Compute eigendecomposition (simplified - power iteration based)
 */
int eigen_decomposition(const Matrix *A, Matrix **eigenvectors, double *eigenvalues, int n_components);


/* --- neural_network.h --- */
/**
 * neural_network.h - Feedforward Neural Network (Multi-Layer Perceptron)
 *
 * Features:
 * - Configurable hidden layers and neurons
 * - Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
 * - Backpropagation with gradient descent
 * - Mini-batch training
 * - Dropout regularization (optional)
 */



/**
 * Activation functions
 */
typedef enum {
    ACTIVATION_IDENTITY,    // f(x) = x
    ACTIVATION_SIGMOID,     // f(x) = 1 / (1 + exp(-x))
    ACTIVATION_TANH,        // f(x) = tanh(x)
    ACTIVATION_RELU,        // f(x) = max(0, x)
    ACTIVATION_LEAKY_RELU,  // f(x) = x if x > 0 else 0.01x
    ACTIVATION_SOFTMAX      // f(x_i) = exp(x_i) / sum(exp(x_j))
} ActivationType;

/**
 * Loss functions
 */
typedef enum {
    LOSS_MSE,              // Mean Squared Error (regression)
    LOSS_CROSS_ENTROPY,    // Cross-entropy (classification)
    LOSS_BINARY_CE         // Binary cross-entropy
} LossType;

/**
 * Optimizer types
 */
typedef enum {
    OPTIMIZER_SGD,         // Stochastic Gradient Descent
    OPTIMIZER_MOMENTUM,    // SGD with momentum
    OPTIMIZER_ADAM         // Adaptive Moment Estimation
} OptimizerType;

/**
 * Layer structure
 */
typedef struct {
    Matrix *weights;        // (n_input x n_output)
    Matrix *biases;         // (1 x n_output)
    ActivationType activation;

    // Gradients
    Matrix *d_weights;
    Matrix *d_biases;

    // For momentum/Adam
    Matrix *v_weights;      // Velocity for weights
    Matrix *v_biases;       // Velocity for biases
    Matrix *m_weights;      // First moment (Adam)
    Matrix *m_biases;       // First moment (Adam)

    // Cache for backprop
    Matrix *input_cache;    // Input to this layer
    Matrix *output_cache;   // Output after activation
    Matrix *z_cache;        // Pre-activation values
} Layer;

/**
 * Neural Network structure
 */
typedef struct {
    Estimator base;

    Layer **layers;
    int n_layers;

    // Architecture
    int *layer_sizes;       // Array of layer sizes
    int n_layer_sizes;

    // Hyperparameters
    double learning_rate;
    int max_epochs;
    int batch_size;
    double momentum;
    double beta1;           // Adam first moment decay
    double beta2;           // Adam second moment decay
    double epsilon;         // Adam epsilon
    double l2_reg;          // L2 regularization
    double dropout_rate;    // Dropout probability

    LossType loss_type;
    OptimizerType optimizer;
    ActivationType hidden_activation;
    ActivationType output_activation;

    // Training state
    int epoch;
    int t;                  // Time step for Adam

    // Classification info
    int n_classes;
    int n_features;
} NeuralNetwork;

/**
 * MLP Classifier (wrapper)
 */
typedef NeuralNetwork MLPClassifier;

/**
 * MLP Regressor (wrapper)
 */
typedef NeuralNetwork MLPRegressor;

/* ============================================
 * Neural Network API
 * ============================================ */

/**
 * Create a simple neural network
 *
 * @param hidden_layer_sizes Array of hidden layer sizes (e.g., [64, 32])
 * @param n_hidden Number of hidden layers
 * @param activation Activation for hidden layers
 * @return Neural network
 */
NeuralNetwork* neural_network_create(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType activation
);

/**
 * Create with full configuration
 */
NeuralNetwork* neural_network_create_full(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType hidden_activation,
    ActivationType output_activation,
    LossType loss,
    OptimizerType optimizer,
    double learning_rate,
    int max_epochs,
    int batch_size,
    double momentum,
    double l2_reg
);

/**
 * Initialize network with input/output dimensions
 */
void neural_network_init(NeuralNetwork *nn, int n_features, int n_outputs);

/**
 * Fit the network
 */
Estimator* neural_network_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Forward pass
 */
Matrix* neural_network_forward(NeuralNetwork *nn, const Matrix *X);

/**
 * Backward pass (compute gradients)
 */
void neural_network_backward(NeuralNetwork *nn, const Matrix *y);

/**
 * Update weights using gradients
 */
void neural_network_update_weights(NeuralNetwork *nn);

/**
 * Predict
 */
Matrix* neural_network_predict(const Estimator *self, const Matrix *X);

/**
 * Predict probabilities (for classification)
 */
Matrix* neural_network_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score
 */
double neural_network_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* neural_network_clone(const Estimator *self);

/**
 * Free
 */
void neural_network_free(Estimator *self);

/**
 * Print network summary
 */
void neural_network_print_summary(const Estimator *self);

/* ============================================
 * MLP Classifier/Regressor API
 * ============================================ */

/**
 * Create MLP Classifier
 */
MLPClassifier* mlp_classifier_create(const int *hidden_layer_sizes, int n_hidden);

/**
 * Create MLP Regressor
 */
MLPRegressor* mlp_regressor_create(const int *hidden_layer_sizes, int n_hidden);

/* ============================================
 * Activation function utilities
 * ============================================ */

/**
 * Apply activation function element-wise
 */
Matrix* activation_forward(const Matrix *z, ActivationType type);

/**
 * Compute activation derivative
 */
Matrix* activation_derivative(const Matrix *z, const Matrix *a, ActivationType type);

/* ============================================
 * Layer utilities
 * ============================================ */

/**
 * Create a layer
 */
Layer* layer_create(int n_input, int n_output, ActivationType activation);

/**
 * Free a layer
 */
void layer_free(Layer *layer);


/* --- validation.h --- */
/**
 * validation.h - Cross-validation and model selection utilities
 *
 * Provides scikit-learn style cross-validation:
 * - k-fold cross-validation
 * - Stratified k-fold (for classification)
 * - Leave-one-out
 * - cross_val_score() function
 */



/**
 * K-Fold split indices
 */
typedef struct {
    size_t *train_indices;
    size_t n_train;
    size_t *test_indices;
    size_t n_test;
} FoldIndices;

/**
 * K-Fold cross-validator
 */
typedef struct {
    int n_splits;           // Number of folds
    int shuffle;            // Whether to shuffle before splitting
    unsigned int seed;      // Random seed for shuffling
    size_t n_samples;       // Total number of samples
    size_t *indices;        // Shuffled indices
    int current_fold;       // Current fold being iterated
} KFold;

/**
 * Stratified K-Fold cross-validator
 */
typedef struct {
    int n_splits;
    int shuffle;
    unsigned int seed;
    size_t n_samples;
    size_t *indices;
    const Matrix *y;        // Target labels for stratification
    int n_classes;          // Number of unique classes
} StratifiedKFold;

/**
 * Cross-validation results
 */
typedef struct {
    double *test_scores;    // Score for each fold
    double *train_scores;   // Training score for each fold (optional)
    double *fit_times;      // Time to fit each fold
    double *score_times;    // Time to score each fold
    int n_splits;           // Number of folds
    double mean_test_score;
    double std_test_score;
    double mean_train_score;
    double std_train_score;
} CrossValResults;

/* ============================================
 * K-Fold Cross-Validation
 * ============================================ */

/**
 * Create a K-Fold cross-validator
 *
 * @param n_splits Number of folds (typically 5 or 10)
 * @param shuffle Whether to shuffle data before splitting
 * @param seed Random seed for shuffling
 * @return KFold structure
 */
KFold* kfold_create(int n_splits, int shuffle, unsigned int seed);

/**
 * Initialize K-Fold with data size
 */
void kfold_init(KFold *kf, size_t n_samples);

/**
 * Get fold indices for a specific fold
 */
FoldIndices kfold_get_fold(const KFold *kf, int fold);

/**
 * Free fold indices
 */
void fold_indices_free(FoldIndices *fi);

/**
 * Free K-Fold
 */
void kfold_free(KFold *kf);

/* ============================================
 * Stratified K-Fold (for classification)
 * ============================================ */

/**
 * Create a Stratified K-Fold cross-validator
 * Ensures each fold has approximately the same class distribution
 */
StratifiedKFold* stratified_kfold_create(int n_splits, int shuffle, unsigned int seed);

/**
 * Initialize with labels
 */
void stratified_kfold_init(StratifiedKFold *skf, const Matrix *y);

/**
 * Get fold indices
 */
FoldIndices stratified_kfold_get_fold(const StratifiedKFold *skf, int fold);

/**
 * Free Stratified K-Fold
 */
void stratified_kfold_free(StratifiedKFold *skf);

/* ============================================
 * Cross-validation scoring
 * ============================================ */

/**
 * Evaluate estimator using cross-validation
 *
 * Similar to sklearn.model_selection.cross_val_score
 *
 * @param estimator Model to evaluate (will be cloned for each fold)
 * @param X Feature matrix
 * @param y Target vector
 * @param n_splits Number of folds
 * @param shuffle Whether to shuffle data
 * @param seed Random seed
 * @return CrossValResults with scores for each fold
 */
CrossValResults* cross_val_score(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
);

/**
 * Cross-validation with stratified folds (for classification)
 */
CrossValResults* cross_val_score_stratified(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
);

/**
 * Cross-validation with custom scoring function
 */
typedef double (*ScoringFunc)(const Matrix *y_true, const Matrix *y_pred);

CrossValResults* cross_val_score_custom(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed,
    ScoringFunc scorer
);

/**
 * Print cross-validation results
 */
void cross_val_results_print(const CrossValResults *results);

/**
 * Free cross-validation results
 */
void cross_val_results_free(CrossValResults *results);

/* ============================================
 * Leave-One-Out Cross-Validation
 * ============================================ */

/**
 * Leave-One-Out cross-validation
 * Each sample is used once as test set
 */
CrossValResults* leave_one_out_cv(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y
);

/* ============================================
 * Train/Test Split utilities
 * ============================================ */

/**
 * Get subset of matrix by row indices
 */
Matrix* matrix_get_rows(const Matrix *m, const size_t *indices, size_t n_indices);


/* --- feature_selection.h --- */
/**
 * feature_selection.h - Feature Selection Utilities
 *
 * Provides:
 * - SelectKBest: Select top k features by score
 * - VarianceThreshold: Remove low-variance features
 * - f_classif / f_regression: ANOVA F-test scoring
 * - mutual_info_classif / mutual_info_regression: Mutual information
 */



/**
 * Feature scoring functions
 */
typedef enum {
    SCORE_F_CLASSIF,         // ANOVA F-value for classification
    SCORE_F_REGRESSION,      // F-value for regression
    SCORE_MUTUAL_INFO_CLASSIF,    // Mutual information (classification)
    SCORE_MUTUAL_INFO_REGRESSION, // Mutual information (regression)
    SCORE_CHI2               // Chi-squared (for non-negative features)
} ScoreFunction;

/**
 * SelectKBest - Select features according to k highest scores
 */
typedef struct {
    Estimator base;

    // Configuration
    ScoreFunction score_func;
    int k;                   // Number of features to select

    // Fitted attributes
    double *scores_;         // Score for each feature
    double *pvalues_;        // p-value for each feature (if applicable)
    int *support_;           // Mask of selected features (1 = selected)
    int n_features_;
    int n_features_selected_;
} SelectKBest;

/**
 * VarianceThreshold - Remove features with variance below threshold
 */
typedef struct {
    Estimator base;

    // Configuration
    double threshold;

    // Fitted attributes
    double *variances_;
    int *support_;
    int n_features_;
    int n_features_selected_;
} VarianceThreshold;

/**
 * RFE - Recursive Feature Elimination
 */
typedef struct {
    Estimator base;

    // Configuration
    Estimator *estimator;    // Estimator with feature_importances or coef_
    int n_features_to_select;
    int step;                // Features to remove per iteration (1 = one at a time)

    // Fitted attributes
    int *support_;           // Mask of selected features
    int *ranking_;           // Feature ranking (1 = best)
    int n_features_;
} RFE;

/* ============================================
 * SelectKBest API
 * ============================================ */

/**
 * Create SelectKBest transformer
 *
 * @param score_func Scoring function to use
 * @param k Number of top features to select (0 = all)
 * @return SelectKBest instance
 */
SelectKBest* select_k_best_create(ScoreFunction score_func, int k);

/**
 * Fit SelectKBest
 */
Estimator* select_k_best_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform - reduce X to selected features
 */
Matrix* select_k_best_transform(const Estimator *self, const Matrix *X);

/**
 * Get feature scores
 */
const double* select_k_best_scores(const SelectKBest *skb);

/**
 * Get feature support mask
 */
const int* select_k_best_get_support(const SelectKBest *skb);

/**
 * Clone
 */
Estimator* select_k_best_clone(const Estimator *self);

/**
 * Free
 */
void select_k_best_free(Estimator *self);

/**
 * Print summary
 */
void select_k_best_print_summary(const Estimator *self);

/* ============================================
 * VarianceThreshold API
 * ============================================ */

/**
 * Create VarianceThreshold transformer
 *
 * @param threshold Features with variance <= threshold are removed
 * @return VarianceThreshold instance
 */
VarianceThreshold* variance_threshold_create(double threshold);

/**
 * Fit VarianceThreshold
 */
Estimator* variance_threshold_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Transform - remove low-variance features
 */
Matrix* variance_threshold_transform(const Estimator *self, const Matrix *X);

/**
 * Get variances
 */
const double* variance_threshold_variances(const VarianceThreshold *vt);

/**
 * Get support mask
 */
const int* variance_threshold_get_support(const VarianceThreshold *vt);

/**
 * Clone
 */
Estimator* variance_threshold_clone(const Estimator *self);

/**
 * Free
 */
void variance_threshold_free(Estimator *self);

/* ============================================
 * Scoring Functions (standalone)
 * ============================================ */

/**
 * Compute ANOVA F-value for classification
 * Returns F-statistic and p-values for each feature
 *
 * @param X Feature matrix (n_samples x n_features)
 * @param y Class labels (n_samples x 1)
 * @param f_values Output: F-statistic for each feature (caller allocates)
 * @param p_values Output: p-value for each feature (caller allocates, can be NULL)
 * @return 0 on success
 */
int f_classif(const Matrix *X, const Matrix *y, double *f_values, double *p_values);

/**
 * Compute F-value for regression (correlation-based)
 */
int f_regression(const Matrix *X, const Matrix *y, double *f_values, double *p_values);

/**
 * Compute chi-squared statistic (for non-negative features)
 */
int chi2(const Matrix *X, const Matrix *y, double *chi2_values, double *p_values);

/**
 * Compute mutual information for classification
 * Uses binning-based estimation
 */
int mutual_info_classif(const Matrix *X, const Matrix *y, double *mi_values, int n_bins);

/**
 * Compute mutual information for regression
 */
int mutual_info_regression(const Matrix *X, const Matrix *y, double *mi_values, int n_bins);

/* ============================================
 * Utility Functions
 * ============================================ */

/**
 * Get feature importances from a fitted estimator
 * Works with: LinearRegression, DecisionTree, RandomForest
 *
 * @param estimator Fitted estimator
 * @param importances Output array (caller allocates n_features)
 * @return 0 on success, -1 if not supported
 */
int get_feature_importances(const Estimator *estimator, double *importances);


/* --- ensemble.h --- */
/**
 * ensemble.h - Ensemble methods
 *
 * Provides:
 * - Random Forest (Classifier and Regressor)
 * - Bagging
 */



/**
 * Random Forest Classifier
 */
typedef struct {
    Estimator base;

    // Configuration
    int n_estimators;           // Number of trees
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;           // Features to consider at each split (0 = sqrt(n_features))
    int bootstrap;              // Whether to use bootstrap samples
    unsigned int seed;

    // Trees
    DecisionTreeClassifier **trees;
    int n_classes;
    int n_features;

    // Out-of-bag score
    double oob_score_;
} RandomForestClassifier;

/**
 * Random Forest Regressor
 */
typedef struct {
    Estimator base;

    int n_estimators;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;
    int bootstrap;
    unsigned int seed;

    DecisionTreeRegressor **trees;
    int n_features;

    double oob_score_;
} RandomForestRegressor;

/* ============================================
 * Random Forest Classifier API
 * ============================================ */

/**
 * Create Random Forest Classifier
 *
 * @param n_estimators Number of trees (default: 100)
 * @return RandomForestClassifier instance
 */
RandomForestClassifier* random_forest_classifier_create(int n_estimators);

/**
 * Create with full configuration
 */
RandomForestClassifier* random_forest_classifier_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
);

/**
 * Fit classifier
 */
Estimator* random_forest_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict class labels
 */
Matrix* random_forest_classifier_predict(const Estimator *self, const Matrix *X);

/**
 * Predict class probabilities (averaged across trees)
 */
Matrix* random_forest_classifier_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score (accuracy)
 */
double random_forest_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Get feature importances
 */
Matrix* random_forest_classifier_feature_importances(const RandomForestClassifier *rf);

/**
 * Clone
 */
Estimator* random_forest_classifier_clone(const Estimator *self);

/**
 * Free
 */
void random_forest_classifier_free(Estimator *self);

/**
 * Print summary
 */
void random_forest_classifier_print_summary(const Estimator *self);

/* ============================================
 * Random Forest Regressor API
 * ============================================ */

/**
 * Create Random Forest Regressor
 */
RandomForestRegressor* random_forest_regressor_create(int n_estimators);

/**
 * Create with full configuration
 */
RandomForestRegressor* random_forest_regressor_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
);

/**
 * Fit regressor
 */
Estimator* random_forest_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Predict values
 */
Matrix* random_forest_regressor_predict(const Estimator *self, const Matrix *X);

/**
 * Score (R²)
 */
double random_forest_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* random_forest_regressor_clone(const Estimator *self);

/**
 * Free
 */
void random_forest_regressor_free(Estimator *self);


/* --- model_selection.h --- */
/**
 * model_selection.h - Model selection and hyperparameter tuning
 *
 * Provides scikit-learn style hyperparameter search:
 * - GridSearchCV: Exhaustive search over parameter grid
 * - RandomizedSearchCV: Random search over parameter distributions
 */



/**
 * Parameter types
 */
typedef enum {
    PARAM_INT,
    PARAM_DOUBLE,
    PARAM_ENUM
} ParamType;

/**
 * Single parameter specification
 */
typedef struct {
    const char *name;
    ParamType type;
    union {
        struct { int *values; int n_values; } int_vals;
        struct { double *values; int n_values; } double_vals;
        struct { int *values; int n_values; } enum_vals;
    } data;
} ParamSpec;

/**
 * Parameter grid
 */
typedef struct {
    ParamSpec *params;
    int n_params;
} ParameterGrid;

/**
 * Grid Search results for a single parameter combination
 */
typedef struct {
    double *param_values;      // Parameter values for this combination
    double mean_test_score;
    double std_test_score;
    double mean_train_score;
    double mean_fit_time;
    int rank;
} GridSearchResult;

/**
 * Grid Search CV structure
 */
typedef struct {
    Estimator *base_estimator;   // Template estimator to clone
    ParameterGrid *param_grid;
    int cv;                       // Number of cross-validation folds
    int shuffle;
    unsigned int seed;
    int verbose;
    int refit;                    // Whether to refit best model on all data

    // Results
    GridSearchResult *results;
    int n_results;
    int best_index;
    double best_score;
    Estimator *best_estimator;   // Refitted best estimator
    double *best_params;

    // Scoring
    int is_classification;
} GridSearchCV;

/* ============================================
 * Parameter Grid API
 * ============================================ */

/**
 * Create a parameter grid
 */
ParameterGrid* param_grid_create(int n_params);

/**
 * Add integer parameter
 */
int param_grid_add_int(ParameterGrid *grid, int index, const char *name,
                       const int *values, int n_values);

/**
 * Add double parameter
 */
int param_grid_add_double(ParameterGrid *grid, int index, const char *name,
                          const double *values, int n_values);

/**
 * Get total number of parameter combinations
 */
int param_grid_get_n_combinations(const ParameterGrid *grid);

/**
 * Get parameter combination by index
 */
double* param_grid_get_combination(const ParameterGrid *grid, int combo_index);

/**
 * Free parameter grid
 */
void param_grid_free(ParameterGrid *grid);

/* ============================================
 * GridSearchCV API
 * ============================================ */

/**
 * Create GridSearchCV
 *
 * @param estimator Base estimator to tune
 * @param param_grid Parameter grid to search
 * @param cv Number of cross-validation folds
 * @return GridSearchCV instance
 */
GridSearchCV* grid_search_cv_create(
    Estimator *estimator,
    ParameterGrid *param_grid,
    int cv
);

/**
 * Fit GridSearchCV - search all parameter combinations
 */
int grid_search_cv_fit(GridSearchCV *gs, const Matrix *X, const Matrix *y);

/**
 * Get best parameters
 */
const double* grid_search_cv_best_params(const GridSearchCV *gs);

/**
 * Get best score
 */
double grid_search_cv_best_score(const GridSearchCV *gs);

/**
 * Get best estimator (refitted on all data)
 */
Estimator* grid_search_cv_best_estimator(GridSearchCV *gs);

/**
 * Predict using best estimator
 */
Matrix* grid_search_cv_predict(const GridSearchCV *gs, const Matrix *X);

/**
 * Score using best estimator
 */
double grid_search_cv_score(const GridSearchCV *gs, const Matrix *X, const Matrix *y);

/**
 * Print CV results
 */
void grid_search_cv_print_results(const GridSearchCV *gs);

/**
 * Free GridSearchCV
 */
void grid_search_cv_free(GridSearchCV *gs);

/* ============================================
 * Learning Curves
 * ============================================ */

/**
 * Learning curve results
 */
typedef struct {
    size_t *train_sizes;
    double *train_scores_mean;
    double *train_scores_std;
    double *test_scores_mean;
    double *test_scores_std;
    int n_points;
} LearningCurveResult;

/**
 * Compute learning curve
 *
 * @param estimator Model to evaluate
 * @param X Feature matrix
 * @param y Target vector
 * @param train_sizes Array of training set sizes (as fractions 0-1 or absolute)
 * @param n_sizes Number of sizes to evaluate
 * @param cv Number of CV folds
 * @return Learning curve results
 */
LearningCurveResult* learning_curve(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    const double *train_sizes,
    int n_sizes,
    int cv
);

/**
 * Print learning curve
 */
void learning_curve_print(const LearningCurveResult *lc);

/**
 * Save learning curve to CSV
 */
int learning_curve_save_csv(const LearningCurveResult *lc, const char *filename);

/**
 * Free learning curve
 */
void learning_curve_free(LearningCurveResult *lc);


/* --- pipeline.h --- */
/**
 * pipeline.h - Pipeline for chaining transformers and estimators
 *
 * Provides scikit-learn style Pipeline:
 * - Chain multiple preprocessing steps with a final estimator
 * - Single fit/predict interface
 * - Automatic data transformation through each step
 */



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



#ifdef CML_IMPLEMENTATION

/* === IMPLEMENTATION === */

/* --- matrix.c --- */
/**
 * @file matrix.c
 * @brief Implementation of matrix operations
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

Matrix* matrix_alloc(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }

    Matrix *m = malloc(sizeof(Matrix));
    if (!m) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));

    if (!m->data) {
        free(m);
        return NULL;
    }

    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_copy(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *copy = matrix_alloc(m->rows, m->cols);
    if (!copy) {
        return NULL;
    }

    memcpy(copy->data, m->data, m->rows * m->cols * sizeof(double));
    return copy;
}

double matrix_get(const Matrix *m, size_t i, size_t j) {
    if (!m || i >= m->rows || j >= m->cols) {
        cml_set_error(CML_ERROR_OUT_OF_BOUNDS,
                      "matrix_get: out of bounds or NULL (%zu, %zu) vs (%zu, %zu)",
                      i, j, m ? m->rows : 0, m ? m->cols : 0);
        return 0.0;  /* Safe default, avoids crash */
    }
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, size_t i, size_t j, double val) {
    if (!m || i >= m->rows || j >= m->cols) {
        cml_set_error(CML_ERROR_OUT_OF_BOUNDS,
                      "matrix_set: out of bounds or NULL (%zu, %zu) vs (%zu, %zu)",
                      i, j, m ? m->rows : 0, m ? m->cols : 0);
        return;  /* No-op on invalid access */
    }
    m->data[i * m->cols + j] = val;
}

Matrix* matrix_add(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_add: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix* matrix_sub(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_sub: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix* matrix_mul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_mul: dimension mismatch (%zu x %zu) vs (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_alloc(a->rows, a->cols);
    if (!result) {
        return NULL;
    }

    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}

Matrix* matrix_scale(const Matrix *m, double scalar) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->rows, m->cols);
    if (!result) {
        return NULL;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        result->data[i] = m->data[i] * scalar;
    }

    return result;
}

#ifdef _OPENMP
static Matrix* matrix_matmul_omp(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        return NULL;
    }

    if (a->cols != b->rows) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_matmul: dimension mismatch (%zu x %zu) * (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    size_t M = a->rows, K = a->cols, N = b->cols;
    Matrix *result = matrix_alloc(M, N);
    if (!result) {
        return NULL;
    }

    /* Allocate transposed B for cache-friendly column access */
    double *b_colmajor = malloc(K * N * sizeof(double));
    if (!b_colmajor) {
        matrix_free(result);
        return NULL;
    }

    /* Transpose B: b_colmajor[j * K + i] = b->data[i * N + j] */
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b_colmajor[j * K + i] = b->data[i * N + j];
        }
    }

    /* i-k-j loop with OpenMP on outer loop */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double a_ik = a->data[i * K + k];
            for (size_t j = 0; j < N; j++) {
                result->data[i * N + j] += a_ik * b_colmajor[j * K + k];
            }
        }
    }

    free(b_colmajor);
    return result;
}
#endif /* _OPENMP */

Matrix* matrix_matmul(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        cml_set_error(CML_ERROR_NULL_PTR, "matrix_matmul: NULL input");
        return NULL;
    }

    if (a->cols != b->rows) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH,
                      "matrix_matmul: dimension mismatch (%zu x %zu) * (%zu x %zu)",
                      a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

#ifdef _OPENMP
    if (omp_get_max_threads() > 1) return matrix_matmul_omp(a, b);
#endif

    size_t M = a->rows, K = a->cols, N = b->cols;
    Matrix *result = matrix_alloc(M, N);
    if (!result) {
        return NULL;
    }

    /* Allocate transposed B for cache-friendly column access */
    double *b_colmajor = malloc(K * N * sizeof(double));
    if (!b_colmajor) {
        matrix_free(result);
        return NULL;
    }

    /* Transpose B: b_colmajor[j * K + i] = b->data[i * N + j] */
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b_colmajor[j * K + i] = b->data[i * N + j];
        }
    }

    /* i-k-j loop: access A row-wise and B^T row-wise (both sequential) */
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double a_ik = a->data[i * K + k];
            for (size_t j = 0; j < N; j++) {
                result->data[i * N + j] += a_ik * b_colmajor[j * K + k];
            }
        }
    }

    free(b_colmajor);
    return result;
}

Matrix* matrix_transpose(const Matrix *m) {
    if (!m) {
        return NULL;
    }

    Matrix *result = matrix_alloc(m->cols, m->rows);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }

    return result;
}

void matrix_print(const Matrix *m) {
    if (!m) {
        printf("(null matrix)\n");
        return;
    }

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        printf("  [");
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void matrix_fill(Matrix *m, double val) {
    if (!m) {
        return;
    }

    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] = val;
    }
}

Matrix* matrix_identity(size_t n) {
    if (n == 0) {
        return NULL;
    }

    Matrix *m = matrix_alloc(n, n);
    if (!m) {
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        m->data[i * n + i] = 1.0;
    }

    return m;
}


/* --- utils.c --- */
/**
 * @file utils.c
 * @brief Implementation of utility functions
 */


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* xoshiro256** state */
static uint64_t rng_state[4];
static int rng_seeded = 0;

static uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    const uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

/* SplitMix64 for seeding from a single value */
static void rng_seed_from_uint(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
    rng_seeded = 1;
}

void rand_seed(unsigned int seed) {
    rng_seed_from_uint((uint64_t)seed);
}

double rand_uniform(void) {
    if (!rng_seeded) {
        rng_seed_from_uint((uint64_t)time(NULL));
    }
    return (double)(xoshiro256ss() >> 11) / (double)(1ULL << 53);
}

double rand_uniform_range(double min, double max) {
    return min + rand_uniform() * (max - min);
}

double rand_normal(void) {
    /* Box-Muller transform */
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    double u, v, s;
    do {
        u = rand_uniform() * 2.0 - 1.0;
        v = rand_uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;

    return u * s;
}

double rand_normal_params(double mean, double std) {
    return mean + std * rand_normal();
}

double mean(const double *data, size_t n) {
    if (n == 0 || !data) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }

    return sum / (double)n;
}

double variance(const double *data, size_t n) {
    if (n <= 1 || !data) {
        return 0.0;
    }

    double m = mean(data, n);
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - m;
        sum += diff * diff;
    }

    return sum / (double)(n - 1);  /* Bessel's correction */
}

double std_dev(const double *data, size_t n) {
    return sqrt(variance(data, n));
}

void shuffle_indices(size_t *indices, size_t n) {
    if (n <= 1 || !indices) {
        return;
    }

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand_uniform() * (i + 1));
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

char* cml_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *dup = malloc(len + 1);
    if (!dup) return NULL;
    memcpy(dup, s, len + 1);
    return dup;
}


/* --- cml_error.c --- */
/**
 * @file cml_error.c
 * @brief Implementation of unified error handling
 */


#include <stdarg.h>

/* Thread-local last error (C11 _Thread_local) */
static _Thread_local CMLError cml_last_error = { CML_OK, "" };

void cml_set_error(CMLStatus code, const char *fmt, ...) {
    cml_last_error.code = code;
    va_list args;
    va_start(args, fmt);
    vsnprintf(cml_last_error.message, sizeof(cml_last_error.message), fmt, args);
    va_end(args);
}

CMLError cml_get_last_error(void) {
    return cml_last_error;
}

const char* cml_status_string(CMLStatus code) {
    switch (code) {
        case CML_OK:                    return "OK";
        case CML_ERROR_NULL_PTR:        return "NULL pointer";
        case CML_ERROR_OUT_OF_BOUNDS:   return "Out of bounds";
        case CML_ERROR_INVALID_ARG:     return "Invalid argument";
        case CML_ERROR_MEMORY:          return "Memory allocation failed";
        case CML_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CML_ERROR_NOT_FITTED:      return "Model not fitted";
        case CML_ERROR_FILE_IO:         return "File I/O error";
        case CML_ERROR_PARSE:           return "Parse error";
        case CML_ERROR_CONVERGENCE:     return "Convergence failure";
        case CML_ERROR_UNSUPPORTED:     return "Unsupported operation";
        default:                        return "Unknown error";
    }
}


/* --- vector.c --- */
/**
 * @file vector.c
 * @brief Implementation of vector operations
 */



#include <math.h>
#include <stdio.h>

double vector_dot(const Matrix *a, const Matrix *b) {
    if (!a || !b) {
        cml_set_error(CML_ERROR_NULL_PTR, "vector_dot: NULL input");
        return 0.0;
    }

    size_t n_a = a->rows * a->cols;
    size_t n_b = b->rows * b->cols;

    if (n_a != n_b) {
        cml_set_error(CML_ERROR_DIMENSION_MISMATCH, "vector_dot: size mismatch (%zu vs %zu)", n_a, n_b);
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n_a; i++) {
        sum += a->data[i] * b->data[i];
    }

    return sum;
}

double vector_norm(const Matrix *v) {
    if (!v) {
        return 0.0;
    }

    double sum = 0.0;
    size_t n = v->rows * v->cols;

    for (size_t i = 0; i < n; i++) {
        sum += v->data[i] * v->data[i];
    }

    return sqrt(sum);
}

Matrix* vector_scale(const Matrix *v, double scalar) {
    return matrix_scale(v, scalar);
}

Matrix* vector_add(const Matrix *a, const Matrix *b) {
    return matrix_add(a, b);
}

Matrix* vector_sub(const Matrix *a, const Matrix *b) {
    return matrix_sub(a, b);
}


/* --- csv.c --- */
/**
 * @file csv.c
 * @brief Implementation of robust CSV loading and saving
 */

#define _POSIX_C_SOURCE 200809L



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define MAX_LINE_LENGTH 65536
#define INITIAL_CAPACITY 256
#define MAX_FIELD_LEN   4096
#define MAX_HEADERS      1024

/* ── Module-level header storage ────────────────────────────────────── */

static char    *g_header_storage[MAX_HEADERS];
static const char *g_header_ptrs[MAX_HEADERS];
static int      g_header_count = 0;

static void free_headers(void) {
    for (int i = 0; i < g_header_count; i++) {
        free(g_header_storage[i]);
        g_header_storage[i] = NULL;
        g_header_ptrs[i] = NULL;
    }
    g_header_count = 0;
}

const char** csv_get_headers(void) {
    return (const char**)g_header_ptrs;
}

int csv_get_header_count(void) {
    return g_header_count;
}

/* ── Utility helpers ────────────────────────────────────────────────── */

static int is_empty_line(const char *line) {
    while (*line) {
        if (!isspace((unsigned char)*line)) return 0;
        line++;
    }
    return 1;
}

static int is_comment_line(const char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    if (*line == '#') return 1;
    if (line[0] == '/' && line[1] == '/') return 1;
    return 0;
}

/* Skip UTF-8 BOM if present, return pointer past it */
static const char* skip_bom(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    if (p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) {
        return s + 3;
    }
    return s;
}

/**
 * Read the entire file content into a malloc'd buffer.
 * Handles BOM skipping at the start.
 */
static char* read_file_contents(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv: cannot open file '%s'", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) { fclose(fp); return NULL; }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t nread = fread(buf, 1, (size_t)sz, fp);
    buf[nread] = '\0';
    fclose(fp);

    /* Skip BOM in the buffer by shifting content */
    const char *after_bom = skip_bom(buf);
    if (after_bom != buf) {
        size_t offset = (size_t)(after_bom - buf);
        memmove(buf, after_bom, nread - offset + 1);
    }
    return buf;
}

/* ── Delimiter auto-detection ───────────────────────────────────────── */

char csv_detect_delimiter(const char *path) {
    char *contents = read_file_contents(path);
    if (!contents) return ',';

    /* Find first non-empty, non-comment line */
    char *line = contents;
    while (*line) {
        /* skip leading whitespace */
        while (*line && isspace((unsigned char)*line)) line++;
        if (*line == '\0') break;
        if (*line == '#' || (line[0] == '/' && line[1] == '/')) {
            /* skip to next line */
            while (*line && *line != '\n') line++;
            if (*line == '\n') line++;
            continue;
        }
        break;
    }

    /* Count candidate delimiters up to newline */
    int commas = 0, tabs = 0, semis = 0;
    const char *p = line;
    while (*p && *p != '\n' && *p != '\r') {
        switch (*p) {
            case ',': commas++; break;
            case '\t': tabs++; break;
            case ';': semis++; break;
        }
        p++;
    }

    free(contents);

    if (tabs > commas && tabs > semis) return '\t';
    if (semis > commas && semis > tabs) return ';';
    return ',';
}

/* ── Robust field parser ────────────────────────────────────────────── */

/**
 * Parse one field starting at *pos in buf.
 * Writes the raw field string into `field` (up to field_cap-1 chars).
 * Advances *pos past the field and delimiter (if any).
 * Returns 1 if a delimiter followed (more fields), 0 if end-of-record.
 */
static int parse_field(const char *buf, size_t *pos, char *field, size_t field_cap,
                       char delim, char quote) {
    size_t fi = 0;
    int hit_delim = 0;

    /* skip leading whitespace (not newlines) */
    while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
           isspace((unsigned char)buf[*pos])) {
        (*pos)++;
    }

    if (buf[*pos] == quote && quote != '\0') {
        /* Quoted field */
        (*pos)++; /* skip opening quote */
        while (buf[*pos]) {
            if (buf[*pos] == quote) {
                /* Peek: double-quote escape? */
                if (buf[*pos + 1] == quote) {
                    if (fi < field_cap - 1) field[fi++] = quote;
                    *pos += 2;
                } else {
                    /* Closing quote */
                    (*pos)++;
                    break;
                }
            } else {
                if (fi < field_cap - 1) field[fi++] = buf[*pos];
                (*pos)++;
            }
        }
        /* skip trailing whitespace before delimiter */
        while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
               isspace((unsigned char)buf[*pos])) {
            (*pos)++;
        }
        if (buf[*pos] == delim) {
            (*pos)++;
            hit_delim = 1;
        }
    } else {
        /* Unquoted field */
        while (buf[*pos] && buf[*pos] != delim &&
               buf[*pos] != '\n' && buf[*pos] != '\r') {
            if (fi < field_cap - 1) field[fi++] = buf[*pos];
            (*pos)++;
        }
        if (buf[*pos] == delim) {
            (*pos)++;
            hit_delim = 1;
        }
    }

    field[fi] = '\0';

    /* Trim trailing whitespace from unquoted field */
    if (fi > 0) {
        /* Only trim if it wasn't a quoted field — but we already handled
           quoted fields above with trailing whitespace skip. For unquoted,
           trim right. */
        size_t end = fi;
        while (end > 0 && isspace((unsigned char)field[end - 1])) {
            field[--end] = '\0';
        }
    }

    return hit_delim;
}

/**
 * Parse one record (row) starting at *pos.
 * Fields are written into the values array starting at `values[row * ncols + col_offset]`.
 * Returns number of columns parsed, or -1 on skip (empty/comment line).
 * Advances *pos past the record.
 */
static int parse_record(const char *buf, size_t *pos, char delim, char quote,
                        double missing, int max_cols, int is_header_row,
                        char **headers_out) {
    size_t start = *pos;

    /* skip leading whitespace */
    while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
           isspace((unsigned char)buf[*pos])) {
        (*pos)++;
    }

    /* Check comment */
    if (buf[*pos] == '#' || (buf[*pos] == '/' && buf[*pos + 1] == '/')) {
        /* skip to end of line */
        while (buf[*pos] && buf[*pos] != '\n') (*pos)++;
        if (buf[*pos] == '\n') (*pos)++;
        return -1;
    }

    /* Check empty line */
    if (buf[*pos] == '\n' || buf[*pos] == '\r' || buf[*pos] == '\0') {
        if (buf[*pos] == '\r') (*pos)++;
        if (buf[*pos] == '\n') (*pos)++;
        return -1;
    }

    /* Rewind to start of content to parse fields */
    *pos = start;

    char field[MAX_FIELD_LEN];
    int col = 0;

    while (1) {
        /* End of record? */
        if (buf[*pos] == '\0' || buf[*pos] == '\n' || buf[*pos] == '\r') {
            /* Previous parse_field consumed the last field and stopped at
               newline — but if we hit newline here without parse_field,
               it means a trailing empty field wasn't consumed.
               This happens when the line is just "\n" but we already checked that.
               Actually, parse_field will consume up to delim or newline.
               If the record ended right at newline without a trailing delim,
               the loop ends naturally. */
            break;
        }

        int hit_delim = parse_field(buf, pos, field, sizeof(field), delim, quote);

        if (is_header_row) {
            /* Store header name */
            if (headers_out && col < MAX_HEADERS) {
                headers_out[col] = strdup(field);
            }
        } else {
            /* We'll parse the double later — just count columns for now.
               The caller handles storing values. */
        }

        col++;

        if (max_cols > 0 && col >= max_cols) {
            /* skip rest of line */
            while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r') (*pos)++;
            break;
        }

        if (!hit_delim) {
            /* End of record */
            break;
        }
    }

    /* Consume newline */
    if (buf[*pos] == '\r') (*pos)++;
    if (buf[*pos] == '\n') (*pos)++;

    return col;
}

/* ── csv_load_ext: robust loader ────────────────────────────────────── */

Matrix* csv_load_ext(const char *path, const CSVOptions *opts) {
    if (!path) {
        cml_set_error(CML_ERROR_NULL_PTR, "csv_load_ext: NULL path");
        return NULL;
    }

    /* Defaults */
    int    has_header       = 0;
    char   delimiter        = ',';
    char   quote            = '"';
    double missing_val      = NAN;
    int    skip_empty_lines = 1;
    int    max_rows         = 0;
    int    max_cols         = 0;

    if (opts) {
        has_header       = opts->has_header;
        delimiter        = opts->delimiter  ? opts->delimiter  : ',';
        quote            = opts->quote;
        missing_val      = opts->missing_val;
        skip_empty_lines = opts->skip_empty_lines;
        max_rows         = opts->max_rows;
        max_cols         = opts->max_cols;
    }

    if (delimiter == 0) {
        delimiter = csv_detect_delimiter(path);
    }

    free_headers();

    char *buf = read_file_contents(path);
    if (!buf) return NULL;

    /* Two-pass: first pass to count rows/cols, second pass to fill data.
       Simpler: collect all values in a dynamic array. */
    size_t capacity = INITIAL_CAPACITY;
    double *values = malloc(capacity * sizeof(double));
    if (!values) {
        free(buf);
        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
        return NULL;
    }

    size_t total_values = 0;
    size_t rows = 0;
    size_t cols = 0;  /* determined from first data row */
    size_t pos = 0;

    while (buf[pos]) {
        /* Skip whitespace before record */
        while (buf[pos] && (buf[pos] == ' ' || buf[pos] == '\t')) pos++;
        if (buf[pos] == '\0') break;

        /* Check for comment or empty line */
        if (buf[pos] == '#' || (buf[pos] == '/' && buf[pos + 1] == '/')) {
            while (buf[pos] && buf[pos] != '\n') pos++;
            if (buf[pos] == '\n') pos++;
            continue;
        }
        if (buf[pos] == '\n' || buf[pos] == '\r') {
            if (buf[pos] == '\r') pos++;
            if (buf[pos] == '\n') pos++;
            continue;
        }

        /* Parse fields of this record */
        char field[MAX_FIELD_LEN];
        size_t row_cols = 0;

        while (1) {
            if (buf[pos] == '\0' || buf[pos] == '\n' || buf[pos] == '\r') {
                /* If this is the very first char after a delim, it's a trailing
                   empty field. But if row_cols == 0, it's an empty line. */
                break;
            }

            int hit_delim = parse_field(buf, &pos, field, sizeof(field), delimiter, quote);

            /* Header row? */
            if (has_header && rows == 0) {
                if ((int)row_cols < MAX_HEADERS) {
                    g_header_storage[row_cols] = strdup(field);
                    g_header_ptrs[row_cols] = g_header_storage[row_cols];
                }
            } else {
                /* Parse value */
                double val = missing_val;
                if (field[0] != '\0') {
                    char *end;
                    val = strtod(field, &end);
                    if (end == field) {
                        val = missing_val; /* not a number */
                    }
                }

                if (total_values >= capacity) {
                    capacity *= 2;
                    double *nv = realloc(values, capacity * sizeof(double));
                    if (!nv) {
                        free(values);
                        free(buf);
                        free_headers();
                        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
                        return NULL;
                    }
                    values = nv;
                }
                values[total_values++] = val;
            }

            row_cols++;

            if (max_cols > 0 && (int)row_cols >= max_cols) {
                /* skip rest of line */
                while (buf[pos] && buf[pos] != '\n' && buf[pos] != '\r') pos++;
                break;
            }

            if (!hit_delim) break;
        }

        /* Consume newline */
        if (buf[pos] == '\r') pos++;
        if (buf[pos] == '\n') pos++;

        if (row_cols == 0) continue; /* empty record */

        if (has_header && rows == 0) {
            /* This was the header row */
            g_header_count = (int)row_cols;
            cols = row_cols;
            rows = 0; /* header doesn't count as data row */
            continue;
        }

        /* Set column count from first data row */
        if (cols == 0) {
            cols = row_cols;
        }

        /* Pad or truncate to match cols */
        if (row_cols < cols) {
            /* Pad with missing_val */
            for (size_t pc = row_cols; pc < cols; pc++) {
                if (total_values >= capacity) {
                    capacity *= 2;
                    double *nv = realloc(values, capacity * sizeof(double));
                    if (!nv) {
                        free(values);
                        free(buf);
                        free_headers();
                        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
                        return NULL;
                    }
                    values = nv;
                }
                values[total_values++] = missing_val;
            }
        } else if (row_cols > cols) {
            /* Truncate: remove excess values from the end */
            total_values -= (row_cols - cols);
        }

        rows++;

        if (max_rows > 0 && (int)rows >= max_rows) break;
    }

    free(buf);

    if (rows == 0 || cols == 0) {
        free(values);
        cml_set_error(CML_ERROR_PARSE, "csv_load_ext: no data found in '%s'", path);
        return NULL;
    }

    Matrix *m = matrix_alloc(rows, cols);
    if (!m) {
        free(values);
        return NULL;
    }
    memcpy(m->data, values, rows * cols * sizeof(double));
    free(values);

    return m;
}

/* ── Original csv_load (backward compatible) ────────────────────────── */

static int count_columns_simple(const char *line) {
    int count = 1;
    while (*line) {
        if (*line == ',') count++;
        line++;
    }
    return count;
}

Matrix* csv_load(const char *filename, int has_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv_load: cannot open file '%s'", filename);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    size_t capacity = INITIAL_CAPACITY;
    size_t rows = 0;
    size_t cols = 0;

    double *values = malloc(capacity * sizeof(double));
    if (!values) {
        fclose(fp);
        cml_set_error(CML_ERROR_MEMORY, "csv_load: memory allocation failed");
        return NULL;
    }

    /* Skip header if present */
    if (has_header) {
        if (!fgets(line, sizeof(line), fp)) {
            free(values);
            fclose(fp);
            cml_set_error(CML_ERROR_PARSE, "csv_load: file is empty");
            return NULL;
        }
    }

    /* Read data lines */
    while (fgets(line, sizeof(line), fp)) {
        if (is_empty_line(line)) continue;

        if (cols == 0) {
            cols = (size_t)count_columns_simple(line);
        }

        char *ptr = line;
        char *end;
        size_t col = 0;

        while (*ptr && col < cols) {
            while (*ptr && isspace((unsigned char)*ptr)) ptr++;

            double val = strtod(ptr, &end);

            if (ptr == end) {
                cml_set_error(CML_ERROR_PARSE, "csv_load: parse error at row %zu, col %zu",
                              rows + 1, col + 1);
                free(values);
                fclose(fp);
                return NULL;
            }

            if (rows * cols + col >= capacity) {
                capacity *= 2;
                double *nv = realloc(values, capacity * sizeof(double));
                if (!nv) {
                    free(values);
                    fclose(fp);
                    cml_set_error(CML_ERROR_MEMORY, "csv_load: memory allocation failed");
                    return NULL;
                }
                values = nv;
            }

            values[rows * cols + col] = val;
            col++;

            ptr = end;
            while (*ptr && (isspace((unsigned char)*ptr) || *ptr == ',')) {
                if (*ptr == ',') { ptr++; break; }
                ptr++;
            }
        }

        rows++;
    }

    fclose(fp);

    if (rows == 0 || cols == 0) {
        free(values);
        cml_set_error(CML_ERROR_PARSE, "csv_load: no data found");
        return NULL;
    }

    Matrix *m = matrix_alloc(rows, cols);
    if (!m) {
        free(values);
        return NULL;
    }

    memcpy(m->data, values, rows * cols * sizeof(double));
    free(values);
    return m;
}

/* ── csv_save ───────────────────────────────────────────────────────── */

int csv_save(const Matrix *m, const char *filename) {
    if (!m || !filename) return -1;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv_save: cannot open file '%s' for writing", filename);
        return -1;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            fprintf(fp, "%.6f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}


/* --- metrics.c --- */
/**
 * @file metrics.c
 * @brief Implementation of evaluation metrics
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double mse(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = y_true->data[i] - y_pred->data[i];
        sum += diff * diff;
    }

    return sum / (double)n;
}

double rmse(const Matrix *y_true, const Matrix *y_pred) {
    return sqrt(mse(y_true, y_pred));
}

double mae(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += fabs(y_true->data[i] - y_pred->data[i]);
    }

    return sum / (double)n;
}

double accuracy(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < n; i++) {
        /* Compare as integers for classification */
        if ((int)y_true->data[i] == (int)y_pred->data[i]) {
            correct++;
        }
    }

    return (double)correct / (double)n;
}

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = {0, 0, 0, 0};

    if (!y_true || !y_pred) {
        return cm;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols) {
        return cm;
    }

    for (size_t i = 0; i < n; i++) {
        int actual = (int)y_true->data[i];
        int predicted = (int)y_pred->data[i];

        if (actual == 1 && predicted == 1) {
            cm.tp++;
        } else if (actual == 0 && predicted == 0) {
            cm.tn++;
        } else if (actual == 0 && predicted == 1) {
            cm.fp++;
        } else if (actual == 1 && predicted == 0) {
            cm.fn++;
        }
    }

    return cm;
}

double precision(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fp;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double recall(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fn;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double f1_score(const Matrix *y_true, const Matrix *y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);

    if (p + r == 0.0) {
        return 0.0;
    }

    return 2.0 * (p * r) / (p + r);
}

void confusion_matrix_print(const ConfusionMatrix *cm) {
    if (!cm) {
        return;
    }

    printf("Confusion Matrix:\n");
    printf("                 Predicted\n");
    printf("              Neg      Pos\n");
    printf("Actual Neg  %5d    %5d\n", cm->tn, cm->fp);
    printf("       Pos  %5d    %5d\n", cm->fn, cm->tp);
    printf("\n");
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n", cm->tp, cm->tn, cm->fp, cm->fn);
}

/* ============================================
 * Multi-class metrics
 * ============================================ */

Matrix* confusion_matrix_multi(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) return NULL;

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) return NULL;

    /* Find max label to determine n_classes */
    int max_label = 0;
    for (size_t i = 0; i < n; i++) {
        int lt = (int)y_true->data[i];
        int lp = (int)y_pred->data[i];
        if (lt > max_label) max_label = lt;
        if (lp > max_label) max_label = lp;
    }
    int nc = max_label + 1;

    Matrix *cm = matrix_alloc((size_t)nc, (size_t)nc);
    if (!cm) return NULL;

    /* Count occurrences: cm[true_class][pred_class] */
    for (size_t i = 0; i < n; i++) {
        int t = (int)y_true->data[i];
        int p = (int)y_pred->data[i];
        if (t >= 0 && t < nc && p >= 0 && p < nc) {
            cm->data[t * (size_t)nc + p] += 1.0;
        }
    }

    return cm;
}

/**
 * Helper: compute per-class precision for class c from multi-class confusion matrix.
 * Precision_c = cm[c][c] / sum over all rows of cm[?][c]  (column sum for c)
 */
static double per_class_precision(const Matrix *cm, int c) {
    int nc = (int)cm->rows;
    double tp = cm->data[c * (size_t)nc + c];
    double col_sum = 0.0;
    for (int i = 0; i < nc; i++) {
        col_sum += cm->data[i * (size_t)nc + c];
    }
    if (col_sum == 0.0) return 0.0;
    return tp / col_sum;
}

/**
 * Helper: compute per-class recall for class c from multi-class confusion matrix.
 * Recall_c = cm[c][c] / sum over all cols of cm[c][?]  (row sum for c)
 */
static double per_class_recall(const Matrix *cm, int c) {
    int nc = (int)cm->cols;
    double tp = cm->data[c * (size_t)nc + c];
    double row_sum = 0.0;
    for (int j = 0; j < nc; j++) {
        row_sum += cm->data[c * (size_t)nc + j];
    }
    if (row_sum == 0.0) return 0.0;
    return tp / row_sum;
}

double precision_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        /* Only include classes that appear in predictions (column sum > 0) */
        double p = per_class_precision(cm, c);
        sum += p;
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double recall_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        double r = per_class_recall(cm, c);
        sum += r;
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double f1_score_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        double p = per_class_precision(cm, c);
        double r = per_class_recall(cm, c);
        if (p + r > 0.0) {
            sum += 2.0 * (p * r) / (p + r);
        }
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double f1_score_weighted(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    size_t n = y_true->rows * y_true->cols;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    /* Compute support (count of each class in y_true) */
    double *support = calloc((size_t)n_classes, sizeof(double));
    if (!support) {
        matrix_free(cm);
        return 0.0;
    }
    for (size_t i = 0; i < n; i++) {
        int label = (int)y_true->data[i];
        if (label >= 0 && label < n_classes) {
            support[label] += 1.0;
        }
    }

    double weighted_sum = 0.0;
    for (int c = 0; c < n_classes; c++) {
        double p = per_class_precision(cm, c);
        double r = per_class_recall(cm, c);
        double f1 = 0.0;
        if (p + r > 0.0) {
            f1 = 2.0 * (p * r) / (p + r);
        }
        weighted_sum += f1 * support[c];
    }

    matrix_free(cm);
    free(support);

    if ((double)n == 0.0) return 0.0;
    return weighted_sum / (double)n;
}

/* ============================================
 * Clustering Metrics
 * ============================================ */

double silhouette_score(const Matrix *X, const Matrix *labels, int n_clusters) {
    if (!X || !labels || n_clusters <= 1) return 0.0;

    size_t n = X->rows;
    size_t p = X->cols;
    size_t nl = labels->rows * labels->cols;
    if (n != nl || n == 0) return 0.0;

    /* Count cluster sizes */
    size_t *cluster_size = calloc((size_t)n_clusters, sizeof(size_t));
    if (!cluster_size) return 0.0;

    for (size_t i = 0; i < n; i++) {
        int c = (int)labels->data[i];
        if (c < 0 || c >= n_clusters) {
            free(cluster_size);
            return 0.0;
        }
        cluster_size[c]++;
    }

    /* Silhouette is undefined if any cluster has < 2 members (a(i)=0 division) */
    for (int c = 0; c < n_clusters; c++) {
        if (cluster_size[c] < 2) {
            free(cluster_size);
            return 0.0;
        }
    }

    double total_sil = 0.0;

    for (size_t i = 0; i < n; i++) {
        int ci = (int)labels->data[i];

        /* Compute pairwise distances from i to all other points */
        double *dist_to_cluster = calloc((size_t)n_clusters, sizeof(double));
        if (!dist_to_cluster) {
            free(cluster_size);
            return 0.0;
        }

        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            int cj = (int)labels->data[j];

            double d = 0.0;
            for (size_t f = 0; f < p; f++) {
                double diff = X->data[i * p + f] - X->data[j * p + f];
                d += diff * diff;
            }
            dist_to_cluster[cj] += sqrt(d);
        }

        /* a(i): mean distance to same cluster (exclude self) */
        double a_i = 0.0;
        if (cluster_size[ci] > 1) {
            a_i = dist_to_cluster[ci] / (double)(cluster_size[ci] - 1);
        }

        /* b(i): min mean distance to other clusters */
        double b_i = 1e300;
        for (int c = 0; c < n_clusters; c++) {
            if (c == ci) continue;
            if (cluster_size[c] == 0) continue;
            double mean_d = dist_to_cluster[c] / (double)cluster_size[c];
            if (mean_d < b_i) b_i = mean_d;
        }

        /* s(i) */
        double s_i = 0.0;
        if (a_i > 0.0 || b_i > 0.0) {
            double denom = a_i > b_i ? a_i : b_i;
            s_i = (b_i - a_i) / denom;
        }

        total_sil += s_i;
        free(dist_to_cluster);
    }

    free(cluster_size);
    return total_sil / (double)n;
}


/* --- kmeans.c --- */
/**
 * @file kmeans.c
 * @brief Implementation of k-Means clustering
 */



#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Compute Euclidean distance between two vectors */
static double euclidean_distance(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* Find nearest centroid for a data point */
static int find_nearest_centroid(const double *point, const Matrix *centroids) {
    int nearest = 0;
    double min_dist = DBL_MAX;

    for (size_t c = 0; c < centroids->rows; c++) {
        double dist = euclidean_distance(point, &centroids->data[c * centroids->cols], centroids->cols);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = (int)c;
        }
    }

    return nearest;
}

KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed) {
    if (!X || k <= 0 || max_iter <= 0) {
        return NULL;
    }

    size_t n = X->rows;
    size_t d = X->cols;

    if ((size_t)k > n) {
        k = (int)n;
    }

    KMeansModel *model = malloc(sizeof(KMeansModel));
    if (!model) {
        return NULL;
    }

    model->k = k;
    model->max_iter = max_iter;
    model->centroids = matrix_alloc(k, d);

    if (!model->centroids) {
        free(model);
        return NULL;
    }

    /* Initialize centroids randomly from data points */
    rand_seed(seed);
    size_t *used = calloc(n, sizeof(size_t));
    if (!used) {
        kmeans_free(model);
        return NULL;
    }

    for (int c = 0; c < k; c++) {
        size_t idx;
        do {
            idx = (size_t)(rand_uniform() * n);
        } while (used[idx] && c > 0);
        used[idx] = 1;

        for (size_t j = 0; j < d; j++) {
            model->centroids->data[c * d + j] = X->data[idx * d + j];
        }
    }
    free(used);

    /* Cluster assignments */
    int *assignments = malloc(n * sizeof(int));
    if (!assignments) {
        kmeans_free(model);
        return NULL;
    }

    /* Lloyd's algorithm */
    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;

        /* Assignment step: assign each point to nearest centroid */
        for (size_t i = 0; i < n; i++) {
            int nearest = find_nearest_centroid(&X->data[i * d], model->centroids);
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = 1;
            }
        }

        /* Check for convergence */
        if (!changed && iter > 0) {
            break;
        }

        /* Update step: recompute centroids as mean of assigned points */
        int *counts = calloc(k, sizeof(int));
        if (!counts) {
            free(assignments);
            kmeans_free(model);
            return NULL;
        }

        /* Reset centroids */
        memset(model->centroids->data, 0, k * d * sizeof(double));

        /* Sum assigned points */
        for (size_t i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            for (size_t j = 0; j < d; j++) {
                model->centroids->data[c * d + j] += X->data[i * d + j];
            }
        }

        /* Compute means */
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < d; j++) {
                    model->centroids->data[c * d + j] /= counts[c];
                }
            }
        }

        free(counts);
    }

    free(assignments);
    return model;
}

Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X) {
    if (!model || !X || X->cols != model->centroids->cols) {
        return NULL;
    }

    Matrix *labels = matrix_alloc(X->rows, 1);
    if (!labels) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        labels->data[i] = find_nearest_centroid(&X->data[i * X->cols], model->centroids);
    }

    return labels;
}

void kmeans_free(KMeansModel *model) {
    if (model) {
        matrix_free(model->centroids);
        free(model);
    }
}


/* --- dbscan.c --- */
/**
 * @file dbscan.c
 * @brief DBSCAN density-based clustering implementation
 *
 * Brute-force O(n^2) approach. Uses BFS for cluster expansion.
 */


#include <stdlib.h>
#include <math.h>
#include <string.h>

/* --- Euclidean distance squared between two rows --- */
static double dist_sq(const Matrix *X, size_t i, size_t j) {
    size_t F = X->cols;
    double sum = 0.0;
    for (size_t d = 0; d < F; d++) {
        double diff = X->data[i * F + d] - X->data[j * F + d];
        sum += diff * diff;
    }
    return sum;
}

/* --- Find neighbors within epsilon, return count and fill indices --- */
static int* find_neighbors(const Matrix *X, size_t point, double eps_sq, int *out_count) {
    size_t N = X->rows;
    int *neighbors = malloc(N * sizeof(int));
    if (!neighbors) { *out_count = 0; return NULL; }

    int count = 0;
    for (size_t j = 0; j < N; j++) {
        if (dist_sq(X, point, j) <= eps_sq) {
            neighbors[count++] = (int)j;
        }
    }
    *out_count = count;
    return neighbors;
}

int* dbscan_fit(const DBSCAN *model, const Matrix *X) {
    if (!model || !X) return NULL;

    size_t N = X->rows;
    double eps_sq = model->epsilon * model->epsilon;

    int *labels = malloc(N * sizeof(int));
    if (!labels) return NULL;

    /* Initialize all labels to -1 (unvisited/noise) */
    for (size_t i = 0; i < N; i++) labels[i] = -1;

    /* Determine core points */
    int *is_core = calloc(N, sizeof(int));
    if (!is_core) { free(labels); return NULL; }

    /* For each point, compute neighbor count to determine core status */
    int **neighbor_lists = malloc(N * sizeof(int*));
    int  *neighbor_counts = malloc(N * sizeof(int));
    if (!neighbor_lists || !neighbor_counts) {
        free(labels); free(is_core);
        if (neighbor_lists) free(neighbor_lists);
        if (neighbor_counts) free(neighbor_counts);
        return NULL;
    }

    for (size_t i = 0; i < N; i++) {
        neighbor_lists[i] = find_neighbors(X, i, eps_sq, &neighbor_counts[i]);
        if (neighbor_counts[i] >= model->min_samples) {
            is_core[i] = 1;
        }
    }

    /* BFS to expand clusters from core points */
    int cluster_id = 0;

    /* BFS queue */
    int *queue = malloc(N * sizeof(int));
    if (!queue) {
        free(labels); free(is_core);
        for (size_t i = 0; i < N; i++) free(neighbor_lists[i]);
        free(neighbor_lists); free(neighbor_counts);
        return NULL;
    }

    for (size_t i = 0; i < N; i++) {
        if (labels[i] != -1) continue;      /* already visited */
        if (!is_core[i]) continue;           /* not a core point, stays noise for now */

        /* Start a new cluster */
        labels[i] = cluster_id;

        int front = 0, back = 0;
        queue[back++] = (int)i;

        while (front < back) {
            int cur = queue[front++];

            /* For each neighbor of current point */
            for (int n = 0; n < neighbor_counts[cur]; n++) {
                int nb = neighbor_lists[cur][n];

                if (labels[nb] == -1) {
                    /* Was noise/unvisited — assign to cluster */
                    labels[nb] = cluster_id;
                    if (is_core[nb]) {
                        queue[back++] = nb;
                    }
                } else if (labels[nb] == -2) {
                    /* Shouldn't happen, but be safe */
                    labels[nb] = cluster_id;
                }
                /* If already labeled (part of current cluster), skip */
            }
        }

        cluster_id++;
    }

    /* Cleanup */
    free(queue);
    for (size_t i = 0; i < N; i++) free(neighbor_lists[i]);
    free(neighbor_lists);
    free(neighbor_counts);
    free(is_core);

    return labels;
}

int dbscan_count_clusters(const int *labels, int n) {
    if (!labels || n <= 0) return 0;

    int max_label = -1;
    for (int i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    /* Clusters are numbered 0..max_label, so count = max_label + 1 */
    return (max_label >= 0) ? max_label + 1 : 0;
}

DBSCAN* dbscan_create(void) {
    DBSCAN *model = calloc(1, sizeof(DBSCAN));
    if (!model) return NULL;
    model->epsilon = 0.5;
    model->min_samples = 5;
    return model;
}

void dbscan_free(DBSCAN *model) {
    if (model) free(model);
}


/* --- preprocessing.c --- */
/**
 * @file preprocessing.c
 * @brief Implementation of data preprocessing functions
 */



#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed) {
    TrainTestSplit split = {NULL, NULL, NULL, NULL};

    if (!X || !y || X->rows != y->rows) {
        return split;
    }

    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        return split;
    }

    size_t n = X->rows;
    size_t n_test = (size_t)(n * test_ratio);
    size_t n_train = n - n_test;

    if (n_train == 0 || n_test == 0) {
        return split;
    }

    /* Create shuffled indices */
    size_t *indices = malloc(n * sizeof(size_t));
    if (!indices) {
        return split;
    }

    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }

    rand_seed(seed);
    shuffle_indices(indices, n);

    /* Allocate matrices */
    split.X_train = matrix_alloc(n_train, X->cols);
    split.X_test = matrix_alloc(n_test, X->cols);
    split.y_train = matrix_alloc(n_train, y->cols);
    split.y_test = matrix_alloc(n_test, y->cols);

    if (!split.X_train || !split.X_test || !split.y_train || !split.y_test) {
        train_test_split_free(&split);
        free(indices);
        return (TrainTestSplit){NULL, NULL, NULL, NULL};
    }

    /* Copy training data */
    for (size_t i = 0; i < n_train; i++) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X->cols; j++) {
            split.X_train->data[i * X->cols + j] = X->data[idx * X->cols + j];
        }
        for (size_t j = 0; j < y->cols; j++) {
            split.y_train->data[i * y->cols + j] = y->data[idx * y->cols + j];
        }
    }

    /* Copy test data */
    for (size_t i = 0; i < n_test; i++) {
        size_t idx = indices[n_train + i];
        for (size_t j = 0; j < X->cols; j++) {
            split.X_test->data[i * X->cols + j] = X->data[idx * X->cols + j];
        }
        for (size_t j = 0; j < y->cols; j++) {
            split.y_test->data[i * y->cols + j] = y->data[idx * y->cols + j];
        }
    }

    free(indices);
    return split;
}

void train_test_split_free(TrainTestSplit *split) {
    if (split) {
        matrix_free(split->X_train);
        matrix_free(split->X_test);
        matrix_free(split->y_train);
        matrix_free(split->y_test);
        split->X_train = NULL;
        split->X_test = NULL;
        split->y_train = NULL;
        split->y_test = NULL;
    }
}

Scaler* standardize_fit(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    Scaler *scaler = malloc(sizeof(Scaler));
    if (!scaler) {
        return NULL;
    }

    scaler->n_features = X->cols;
    scaler->means = malloc(X->cols * sizeof(double));
    scaler->stds = malloc(X->cols * sizeof(double));

    if (!scaler->means || !scaler->stds) {
        scaler_free(scaler);
        return NULL;
    }

    /* Compute mean and std for each column */
    for (size_t j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            sum += X->data[i * X->cols + j];
        }
        scaler->means[j] = sum / X->rows;

        double var_sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            double diff = X->data[i * X->cols + j] - scaler->means[j];
            var_sum += diff * diff;
        }
        scaler->stds[j] = sqrt(var_sum / X->rows);

        /* Avoid division by zero */
        if (scaler->stds[j] < DBL_EPSILON) {
            scaler->stds[j] = 1.0;
        }
    }

    return scaler;
}

Matrix* standardize_transform(const Matrix *X, const Scaler *scaler) {
    if (!X || !scaler || X->cols != scaler->n_features) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        for (size_t j = 0; j < X->cols; j++) {
            result->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - scaler->means[j]) / scaler->stds[j];
        }
    }

    return result;
}

Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out) {
    Scaler *scaler = standardize_fit(X);
    if (!scaler) {
        return NULL;
    }

    Matrix *result = standardize_transform(X, scaler);
    if (!result) {
        scaler_free(scaler);
        return NULL;
    }

    if (scaler_out) {
        *scaler_out = scaler;
    } else {
        scaler_free(scaler);
    }

    return result;
}

void scaler_free(Scaler *scaler) {
    if (scaler) {
        free(scaler->means);
        free(scaler->stds);
        free(scaler);
    }
}

MinMaxScaler* minmax_fit(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    MinMaxScaler *scaler = malloc(sizeof(MinMaxScaler));
    if (!scaler) {
        return NULL;
    }

    scaler->n_features = X->cols;
    scaler->mins = malloc(X->cols * sizeof(double));
    scaler->maxs = malloc(X->cols * sizeof(double));

    if (!scaler->mins || !scaler->maxs) {
        minmax_scaler_free(scaler);
        return NULL;
    }

    /* Find min and max for each column */
    for (size_t j = 0; j < X->cols; j++) {
        scaler->mins[j] = DBL_MAX;
        scaler->maxs[j] = -DBL_MAX;

        for (size_t i = 0; i < X->rows; i++) {
            double val = X->data[i * X->cols + j];
            if (val < scaler->mins[j]) {
                scaler->mins[j] = val;
            }
            if (val > scaler->maxs[j]) {
                scaler->maxs[j] = val;
            }
        }

        /* Avoid division by zero */
        if (scaler->maxs[j] - scaler->mins[j] < DBL_EPSILON) {
            scaler->maxs[j] = scaler->mins[j] + 1.0;
        }
    }

    return scaler;
}

Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler) {
    if (!X || !scaler || X->cols != scaler->n_features) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        for (size_t j = 0; j < X->cols; j++) {
            double range = scaler->maxs[j] - scaler->mins[j];
            result->data[i * X->cols + j] =
                (X->data[i * X->cols + j] - scaler->mins[j]) / range;
        }
    }

    return result;
}

void minmax_scaler_free(MinMaxScaler *scaler) {
    if (scaler) {
        free(scaler->mins);
        free(scaler->maxs);
        free(scaler);
    }
}

Matrix* add_bias_column(const Matrix *X) {
    if (!X) {
        return NULL;
    }

    Matrix *result = matrix_alloc(X->rows, X->cols + 1);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < X->rows; i++) {
        result->data[i * result->cols] = 1.0;  /* Bias column */
        for (size_t j = 0; j < X->cols; j++) {
            result->data[i * result->cols + j + 1] = X->data[i * X->cols + j];
        }
    }

    return result;
}


/* --- knn.c --- */
/**
 * @file knn.c
 * @brief Implementation of k-Nearest Neighbors classifier
 */


#include <stdlib.h>
#include <math.h>
#include <float.h>

/* Structure for sorting neighbors by distance */
typedef struct {
    double distance;
    double label;
} Neighbor;

/* Compare function for qsort */
static int compare_neighbors(const void *a, const void *b) {
    double da = ((const Neighbor *)a)->distance;
    double db = ((const Neighbor *)b)->distance;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Compute Euclidean distance between two rows */
static double knn_euclidean_distance(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k) {
    if (!X || !y || X->rows != y->rows || k <= 0) {
        return NULL;
    }

    KNNModel *model = malloc(sizeof(KNNModel));
    if (!model) {
        return NULL;
    }

    model->X_train = matrix_copy(X);
    model->y_train = matrix_copy(y);
    model->k = k;

    if (!model->X_train || !model->y_train) {
        knn_free(model);
        return NULL;
    }

    return model;
}

Matrix* knn_predict(const KNNModel *model, const Matrix *X) {
    if (!model || !X || X->cols != model->X_train->cols) {
        return NULL;
    }

    size_t n_train = model->X_train->rows;
    size_t n_test = X->rows;
    size_t n_features = X->cols;
    int k = model->k;

    /* Limit k to training set size */
    if ((size_t)k > n_train) {
        k = (int)n_train;
    }

    Matrix *predictions = matrix_alloc(n_test, 1);
    if (!predictions) {
        return NULL;
    }

    Neighbor *neighbors = malloc(n_train * sizeof(Neighbor));
    if (!neighbors) {
        matrix_free(predictions);
        return NULL;
    }

    /* For each test sample */
    for (size_t i = 0; i < n_test; i++) {
        /* Compute distances to all training samples */
        for (size_t j = 0; j < n_train; j++) {
            neighbors[j].distance = knn_euclidean_distance(
                &X->data[i * n_features],
                &model->X_train->data[j * n_features],
                n_features
            );
            neighbors[j].label = model->y_train->data[j];
        }

        /* Sort by distance */
        qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

        /* Majority vote among k nearest neighbors */
        /* Simple approach: count votes for each class */
        /* Assuming binary classification or small number of classes */
        double max_label = 0;
        for (size_t j = 0; j < (size_t)k; j++) {
            if (neighbors[j].label > max_label) {
                max_label = neighbors[j].label;
            }
        }

        /* Count votes for each class (0 to max_label) */
        int n_classes = (int)max_label + 1;
        int *votes = calloc(n_classes, sizeof(int));
        if (!votes) {
            free(neighbors);
            matrix_free(predictions);
            return NULL;
        }

        for (int j = 0; j < k; j++) {
            int class_idx = (int)neighbors[j].label;
            if (class_idx >= 0 && class_idx < n_classes) {
                votes[class_idx]++;
            }
        }

        /* Find majority class */
        int max_votes = 0;
        int predicted_class = 0;
        for (int c = 0; c < n_classes; c++) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                predicted_class = c;
            }
        }

        predictions->data[i] = (double)predicted_class;
        free(votes);
    }

    free(neighbors);
    return predictions;
}

Matrix* knn_predict_proba(const KNNModel *model, const Matrix *X, int n_classes) {
    if (!model || !X || X->cols != model->X_train->cols || n_classes <= 0) {
        return NULL;
    }

    size_t n_train = model->X_train->rows;
    size_t n_test = X->rows;
    size_t n_features = X->cols;
    int k = model->k;

    /* Limit k to training set size */
    if ((size_t)k > n_train) {
        k = (int)n_train;
    }

    Matrix *proba = matrix_alloc(n_test, (size_t)n_classes);
    if (!proba) return NULL;

    Neighbor *neighbors = malloc(n_train * sizeof(Neighbor));
    if (!neighbors) {
        matrix_free(proba);
        return NULL;
    }

    /* For each test sample */
    for (size_t i = 0; i < n_test; i++) {
        /* Compute distances to all training samples */
        for (size_t j = 0; j < n_train; j++) {
            neighbors[j].distance = knn_euclidean_distance(
                &X->data[i * n_features],
                &model->X_train->data[j * n_features],
                n_features
            );
            neighbors[j].label = model->y_train->data[j];
        }

        /* Sort by distance */
        qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

        /* Count class frequencies among k nearest neighbors */
        int *votes = calloc((size_t)n_classes, sizeof(int));
        if (!votes) {
            free(neighbors);
            matrix_free(proba);
            return NULL;
        }

        for (int j = 0; j < k; j++) {
            int class_idx = (int)neighbors[j].label;
            if (class_idx >= 0 && class_idx < n_classes) {
                votes[class_idx]++;
            }
        }

        /* Probability = class_count / k */
        for (int c = 0; c < n_classes; c++) {
            proba->data[i * (size_t)n_classes + c] = (double)votes[c] / (double)k;
        }

        free(votes);
    }

    free(neighbors);
    return proba;
}

void knn_free(KNNModel *model) {
    if (model) {
        matrix_free(model->X_train);
        matrix_free(model->y_train);
        free(model);
    }
}


/* --- onehot_encoder.c --- */
/**
 * @file onehot_encoder.c
 * @brief One-hot encoding transformer implementation
 */


#include <stdlib.h>
#include <string.h>

/* ============================================
 * Public API
 * ============================================ */

OneHotEncoder* onehot_encoder_create(void) {
    OneHotEncoder *enc = calloc(1, sizeof(OneHotEncoder));
    if (!enc) return NULL;

    enc->n_features       = 0;
    enc->n_categories     = NULL;
    enc->categories_      = NULL;
    enc->total_output_cols = 0;
    enc->is_fitted        = 0;

    return enc;
}

/* ============================================
 * Internal helpers
 * ============================================ */

static void encoder_free_params(OneHotEncoder *enc) {
    if (!enc) return;
    if (enc->n_categories) {
        free(enc->n_categories);
        enc->n_categories = NULL;
    }
    if (enc->categories_) {
        for (int j = 0; j < enc->n_features; j++) {
            free(enc->categories_[j]);
        }
        free(enc->categories_);
        enc->categories_ = NULL;
    }
    enc->n_features = 0;
    enc->total_output_cols = 0;
    enc->is_fitted = 0;
}

/**
 * Collect unique integer values from a column and sort them.
 * Returns the count of unique values and fills *out_categories.
 */
static int collect_unique_categories(const Matrix *X, size_t col,
                                      int **out_categories, int *out_count) {
    size_t N = X->rows;
    int *values = malloc(N * sizeof(int));
    if (!values) return -1;

    /* Collect all values */
    for (size_t i = 0; i < N; i++) {
        values[i] = (int)X->data[i * X->cols + col];
    }

    /* Find unique values (simple O(n^2) approach) */
    int *unique = malloc(N * sizeof(int));
    if (!unique) {
        free(values);
        return -1;
    }
    int n_unique = 0;

    for (size_t i = 0; i < N; i++) {
        int found = 0;
        for (int k = 0; k < n_unique; k++) {
            if (unique[k] == values[i]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            unique[n_unique++] = values[i];
        }
    }

    /* Sort unique values (simple insertion sort — small arrays) */
    for (int i = 1; i < n_unique; i++) {
        int key = unique[i];
        int j = i - 1;
        while (j >= 0 && unique[j] > key) {
            unique[j + 1] = unique[j];
            j--;
        }
        unique[j + 1] = key;
    }

    /* Allocate exact-sized output */
    *out_categories = malloc(n_unique * sizeof(int));
    if (!*out_categories) {
        free(values);
        free(unique);
        return -1;
    }
    memcpy(*out_categories, unique, n_unique * sizeof(int));
    *out_count = n_unique;

    free(values);
    free(unique);
    return 0;
}

/* ============================================
 * fit
 * ============================================ */

int onehot_encoder_fit(OneHotEncoder *encoder, const Matrix *X) {
    if (!encoder || !X) return -1;

    /* Free previous state */
    encoder_free_params(encoder);

    int F = (int)X->cols;
    encoder->n_features = F;
    encoder->n_categories = calloc(F, sizeof(int));
    encoder->categories_  = calloc(F, sizeof(int*));

    if (!encoder->n_categories || !encoder->categories_) {
        encoder_free_params(encoder);
        return -1;
    }

    encoder->total_output_cols = 0;

    for (int j = 0; j < F; j++) {
        if (collect_unique_categories(X, (size_t)j,
                                       &encoder->categories_[j],
                                       &encoder->n_categories[j]) != 0) {
            encoder_free_params(encoder);
            return -1;
        }
        encoder->total_output_cols += encoder->n_categories[j];
    }

    encoder->is_fitted = 1;
    return 0;
}

/* ============================================
 * transform
 * ============================================ */

Matrix* onehot_encoder_transform(const OneHotEncoder *encoder, const Matrix *X) {
    if (!encoder || !X || !encoder->is_fitted) return NULL;
    if ((int)X->cols != encoder->n_features) return NULL;

    size_t N = X->rows;
    int total_cols = encoder->total_output_cols;

    Matrix *result = matrix_alloc(N, (size_t)total_cols);
    if (!result) return NULL;

    /* Initialize all to 0.0 (matrix_alloc uses calloc-style zero init via its implementation) */
    /* matrix_alloc may or may not zero-init — explicitly fill */
    for (size_t i = 0; i < N * (size_t)total_cols; i++) {
        result->data[i] = 0.0;
    }

    for (size_t i = 0; i < N; i++) {
        int col_offset = 0;
        for (int j = 0; j < encoder->n_features; j++) {
            int val = (int)X->data[i * X->cols + j];
            int n_cats = encoder->n_categories[j];

            /* Find the index of this value in the category list */
            int found_idx = -1;
            for (int k = 0; k < n_cats; k++) {
                if (encoder->categories_[j][k] == val) {
                    found_idx = k;
                    break;
                }
            }

            if (found_idx >= 0) {
                result->data[i * (size_t)total_cols + col_offset + found_idx] = 1.0;
            }

            col_offset += n_cats;
        }
    }

    return result;
}

/* ============================================
 * free
 * ============================================ */

void onehot_encoder_free(OneHotEncoder *encoder) {
    if (encoder) {
        encoder_free_params(encoder);
        free(encoder);
    }
}


/* --- estimator.c --- */
/**
 * estimator.c - Unified Estimator API implementation
 */




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Training history management
 */

TrainingHistory* training_history_alloc(size_t initial_capacity) {
    TrainingHistory *history = malloc(sizeof(TrainingHistory));
    if (!history) return NULL;

    history->loss_history = malloc(initial_capacity * sizeof(double));
    history->metric_history = malloc(initial_capacity * sizeof(double));

    if (!history->loss_history || !history->metric_history) {
        free(history->loss_history);
        free(history->metric_history);
        free(history);
        return NULL;
    }

    history->n_epochs = 0;
    history->capacity = initial_capacity;
    history->converged = 0;
    history->best_epoch = 0;

    return history;
}

void training_history_append(TrainingHistory *history, double loss, double metric) {
    if (!history) return;

    // Expand if needed
    if (history->n_epochs >= history->capacity) {
        size_t new_capacity = history->capacity * 2;
        double *new_loss = realloc(history->loss_history, new_capacity * sizeof(double));
        double *new_metric = realloc(history->metric_history, new_capacity * sizeof(double));

        if (!new_loss || !new_metric) {
            // Keep old arrays if realloc fails
            return;
        }

        history->loss_history = new_loss;
        history->metric_history = new_metric;
        history->capacity = new_capacity;
    }

    history->loss_history[history->n_epochs] = loss;
    history->metric_history[history->n_epochs] = metric;
    history->n_epochs++;
}

void training_history_free(TrainingHistory *history) {
    if (!history) return;
    free(history->loss_history);
    free(history->metric_history);
    free(history);
}

void training_history_print(const TrainingHistory *history) {
    if (!history) {
        printf("No training history available.\n");
        return;
    }

    printf("Training History (%zu epochs):\n", history->n_epochs);
    printf("%-10s %-15s %-15s\n", "Epoch", "Loss", "Metric");
    printf("----------------------------------------\n");

    // Print first 5, last 5 if too many
    size_t print_limit = 10;
    if (history->n_epochs <= print_limit) {
        for (size_t i = 0; i < history->n_epochs; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
    } else {
        for (size_t i = 0; i < 5; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
        printf("...        ...            ...\n");
        for (size_t i = history->n_epochs - 5; i < history->n_epochs; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
    }

    printf("----------------------------------------\n");
    if (history->converged) {
        printf("Converged at epoch %zu\n", history->best_epoch + 1);
    }
    printf("Final loss: %.6f, Final metric: %.6f\n",
           history->loss_history[history->n_epochs - 1],
           history->metric_history[history->n_epochs - 1]);
}

int training_history_save_csv(const TrainingHistory *history, const char *filename) {
    if (!history || !filename) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "epoch,loss,metric\n");
    for (size_t i = 0; i < history->n_epochs; i++) {
        fprintf(f, "%zu,%.10f,%.10f\n",
                i + 1, history->loss_history[i], history->metric_history[i]);
    }

    fclose(f);
    return 0;
}

/**
 * Estimator utilities
 */

int estimator_check_fitted(const Estimator *est, const char *method_name) {
    if (!est) {
        cml_set_error(CML_ERROR_NULL_PTR, "NULL estimator passed to %s", method_name);
        return 0;
    }
    if (!est->is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "Estimator not fitted. Call fit() before %s", method_name);
        return 0;
    }
    return 1;
}

void estimator_set_verbose(Estimator *est, VerboseLevel level) {
    if (est) est->verbose = level;
}

void estimator_set_callback(Estimator *est, TrainingCallback cb, void *user_data) {
    if (est) {
        est->callback = cb;
        est->callback_data = user_data;
    }
}

const TrainingHistory* estimator_get_history(const Estimator *est) {
    return est ? est->history : NULL;
}

Estimator* estimator_clone(const Estimator *est) {
    if (!est || !est->clone) return NULL;
    return est->clone(est);
}

/**
 * Default score implementations
 */

double regression_score_r2(const Estimator *est, const Matrix *X, const Matrix *y) {
    if (!estimator_check_fitted(est, "score")) return -1.0;
    if (!est->predict) return -1.0;

    Matrix *y_pred = est->predict(est, X);
    if (!y_pred) return -1.0;

    // R² = 1 - SS_res / SS_tot
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double y_mean = 0.0;

    // Calculate mean
    for (size_t i = 0; i < y->rows; i++) {
        y_mean += matrix_get(y, i, 0);
    }
    y_mean /= y->rows;

    // Calculate R²
    for (size_t i = 0; i < y->rows; i++) {
        double y_true = matrix_get(y, i, 0);
        double y_hat = matrix_get(y_pred, i, 0);
        ss_res += (y_true - y_hat) * (y_true - y_hat);
        ss_tot += (y_true - y_mean) * (y_true - y_mean);
    }

    matrix_free(y_pred);

    if (ss_tot == 0.0) return 0.0;  // Avoid division by zero
    return 1.0 - (ss_res / ss_tot);
}

double classification_score_accuracy(const Estimator *est, const Matrix *X, const Matrix *y) {
    if (!estimator_check_fitted(est, "score")) return -1.0;
    if (!est->predict) return -1.0;

    Matrix *y_pred = est->predict(est, X);
    if (!y_pred) return -1.0;

    double acc = accuracy(y, y_pred);
    matrix_free(y_pred);

    return acc;
}

/**
 * Verbose output helpers
 */

void verbose_print_epoch(VerboseLevel level, int epoch, int total_epochs,
                         double loss, double metric, const char *metric_name) {
    if (level == VERBOSE_SILENT) return;

    if (level == VERBOSE_DETAILED) {
        printf("Epoch %d/%d - loss: %.6f - %s: %.6f\n",
               epoch, total_epochs, loss, metric_name, metric);
    } else if (level == VERBOSE_PROGRESS) {
        // Print every 10% or every 100 epochs, whichever is smaller
        int interval = total_epochs / 10;
        if (interval < 1) interval = 1;
        if (interval > 100) interval = 100;

        if (epoch % interval == 0 || epoch == total_epochs) {
            printf("Epoch %d/%d - loss: %.6f - %s: %.6f\n",
                   epoch, total_epochs, loss, metric_name, metric);
        }
    }
}

void verbose_print_final(VerboseLevel level, const char *model_name,
                         double final_metric, const char *metric_name, double elapsed_time) {
    if (level == VERBOSE_SILENT) return;

    printf("\n%s trained in %.3f seconds\n", model_name, elapsed_time);
    printf("Final %s: %.6f\n", metric_name, final_metric);
}

void verbose_print_progress_bar(int current, int total, int bar_width) {
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);

    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100));
    fflush(stdout);

    if (current == total) printf("\n");
}


/* --- cml_serialization.c --- */
/**
 * cml_serialization.c - Shared serialization helpers
 */


#include <string.h>

static const char cml_ser_magic[4] = {'C', 'M', 'L', '\0'};

int cml_ser_write_header(FILE *f, int subtype) {
    if (fwrite(cml_ser_magic, 1, CML_SER_MAGIC_SIZE, f) != CML_SER_MAGIC_SIZE) return -1;
    if (fwrite(&subtype, sizeof(int), 1, f) != 1) return -1;
    return 0;
}

int cml_ser_check_header(FILE *f, int expected_subtype) {
    char magic[4];
    if (fread(magic, 1, CML_SER_MAGIC_SIZE, f) != CML_SER_MAGIC_SIZE) return -1;
    if (memcmp(magic, cml_ser_magic, 4) != 0) return -1;
    int subtype;
    if (fread(&subtype, sizeof(int), 1, f) != 1) return -1;
    if (subtype != expected_subtype) return -1;
    return 0;
}

int cml_ser_write_matrix(FILE *f, const Matrix *m) {
    int rows = (int)m->rows;
    int cols = (int)m->cols;
    if (fwrite(&rows, sizeof(int), 1, f) != 1) return -1;
    if (fwrite(&cols, sizeof(int), 1, f) != 1) return -1;
    size_t total = m->rows * m->cols;
    if (fwrite(m->data, sizeof(double), total, f) != total) return -1;
    return 0;
}

Matrix* cml_ser_read_matrix(FILE *f) {
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, f) != 1) return NULL;
    if (fread(&cols, sizeof(int), 1, f) != 1) return NULL;
    Matrix *m = matrix_alloc((size_t)rows, (size_t)cols);
    if (!m) return NULL;
    size_t total = (size_t)rows * (size_t)cols;
    if (fread(m->data, sizeof(double), total, f) != total) {
        matrix_free(m);
        return NULL;
    }
    return m;
}

int cml_ser_write_doubles(FILE *f, const double *data, int n) {
    return fwrite(data, sizeof(double), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_read_doubles(FILE *f, double *data, int n) {
    return fread(data, sizeof(double), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_write_ints(FILE *f, const int *data, int n) {
    return fwrite(data, sizeof(int), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_read_ints(FILE *f, int *data, int n) {
    return fread(data, sizeof(int), (size_t)n, f) == (size_t)n ? 0 : -1;
}

int cml_ser_write_double(FILE *f, double val) {
    return fwrite(&val, sizeof(double), 1, f) == 1 ? 0 : -1;
}

int cml_ser_read_double(FILE *f, double *val) {
    return fread(val, sizeof(double), 1, f) == 1 ? 0 : -1;
}

int cml_ser_write_int(FILE *f, int val) {
    return fwrite(&val, sizeof(int), 1, f) == 1 ? 0 : -1;
}

int cml_ser_read_int(FILE *f, int *val) {
    return fread(val, sizeof(int), 1, f) == 1 ? 0 : -1;
}


/* --- linear_regression.c --- */
/**
 * @file linear_regression.c
 * @brief Implementation of linear regression with Estimator API
 */





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


/* --- logistic_regression.c --- */
/**
 * @file logistic_regression.c
 * @brief Implementation of logistic regression
 */




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


/* --- ridge.c --- */
/**
 * @file ridge.c
 * @brief Ridge (L2-regularized) regression – closed-form solution
 *
 * w = (X^T X + alpha * I)^{-1} X^T y
 * Bias is handled by centering X and y, fitting, then restoring.
 */


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


/* --- lasso.c --- */
/**
 * @file lasso.c
 * @brief Lasso (L1-regularized) regression – coordinate descent
 */


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


/* --- svm.c --- */
/**
 * @file svm.c
 * @brief Linear SVM classifier implementation using sub-gradient descent
 */






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


/* --- naive_bayes.c --- */
/**
 * @file naive_bayes.c
 * @brief Gaussian Naive Bayes classifier implementation
 */





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


/* --- decision_tree.c --- */
/**
 * decision_tree.c - Decision Tree implementation (CART algorithm)
 */





#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ============================================
 * Helper functions
 * ============================================ */

static double gini_impurity(const double *y, const size_t *indices, size_t n, int n_classes) {
    if (n == 0) return 0.0;

    // Count each class
    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 1.0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    // Gini = 1 - sum(p_i^2)
    double gini = 1.0;
    for (int c = 0; c < n_classes; c++) {
        double p = (double)counts[c] / n;
        gini -= p * p;
    }

    free(counts);
    return gini;
}

static double entropy_impurity(const double *y, const size_t *indices, size_t n, int n_classes) {
    if (n == 0) return 0.0;

    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 0.0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    double ent = 0.0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] > 0) {
            double p = (double)counts[c] / n;
            ent -= p * log2(p);
        }
    }

    free(counts);
    return ent;
}

static double mse_impurity(const double *y, const size_t *indices, size_t n) {
    if (n == 0) return 0.0;

    // Calculate mean
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += y[indices[i]];
    }
    mean /= n;

    // Calculate MSE
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = y[indices[i]] - mean;
        mse += diff * diff;
    }

    return mse / n;
}

static int majority_class(const double *y, const size_t *indices, size_t n, int n_classes) {
    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) return 0;

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    int max_count = 0;
    int max_class = 0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] > max_count) {
            max_count = counts[c];
            max_class = c;
        }
    }

    free(counts);
    return max_class;
}

static double mean_value(const double *y, const size_t *indices, size_t n) {
    if (n == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += y[indices[i]];
    }
    return sum / n;
}

static double* class_probabilities(const double *y, const size_t *indices, size_t n, int n_classes) {
    double *proba = calloc(n_classes, sizeof(double));
    if (!proba || n == 0) return proba;

    int *counts = calloc(n_classes, sizeof(int));
    if (!counts) {
        free(proba);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        int label = (int)y[indices[i]];
        if (label >= 0 && label < n_classes) {
            counts[label]++;
        }
    }

    for (int c = 0; c < n_classes; c++) {
        proba[c] = (double)counts[c] / n;
    }

    free(counts);
    return proba;
}

/* ============================================
 * Tree building (Classification)
 * ============================================ */

typedef struct {
    size_t feature_index;
    double threshold;
    double impurity_decrease;
    size_t *left_indices;
    size_t n_left;
    size_t *right_indices;
    size_t n_right;
} BestSplit;

static BestSplit find_best_split_classification(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int n_classes,
    SplitCriterion criterion
) {
    BestSplit best = {0, 0.0, -DBL_MAX, NULL, 0, NULL, 0};

    double (*impurity_func)(const double*, const size_t*, size_t, int);
    if (criterion == CRITERION_ENTROPY) {
        impurity_func = entropy_impurity;
    } else {
        impurity_func = gini_impurity;
    }

    double parent_impurity = impurity_func(y, indices, n, n_classes);

    // Try each feature
    for (size_t feat = 0; feat < X->cols; feat++) {
        // Get unique values and sort
        double *values = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            values[i] = X->data[indices[i] * X->cols + feat];
        }

        // Sort values for threshold search
        qsort(values, n, sizeof(double), compare_doubles);

        // Pre-allocate split buffers outside the threshold loop
        size_t *left = malloc(n * sizeof(size_t));
        size_t *right = malloc(n * sizeof(size_t));
        if (!left || !right) {
            free(values);
            free(left);
            free(right);
            continue;
        }

        // Try midpoints as thresholds
        for (size_t t = 0; t < n - 1; t++) {
            if (values[t] == values[t + 1]) continue;  // Skip duplicates

            double threshold = (values[t] + values[t + 1]) / 2.0;

            // Split indices (reuse pre-allocated buffers)
            size_t n_left = 0, n_right = 0;

            for (size_t i = 0; i < n; i++) {
                double val = X->data[indices[i] * X->cols + feat];
                if (val <= threshold) {
                    left[n_left++] = indices[i];
                } else {
                    right[n_right++] = indices[i];
                }
            }

            if (n_left == 0 || n_right == 0) {
                continue;
            }

            // Calculate impurity decrease
            double left_impurity = impurity_func(y, left, n_left, n_classes);
            double right_impurity = impurity_func(y, right, n_right, n_classes);
            double weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n;
            double impurity_decrease = parent_impurity - weighted_impurity;

            if (impurity_decrease > best.impurity_decrease) {
                free(best.left_indices);
                free(best.right_indices);

                best.feature_index = feat;
                best.threshold = threshold;
                best.impurity_decrease = impurity_decrease;
                best.n_left = n_left;
                best.n_right = n_right;

                // Allocate new buffers for best, reuse left/right for next iteration
                best.left_indices = malloc(n_left * sizeof(size_t));
                best.right_indices = malloc(n_right * sizeof(size_t));
                if (best.left_indices) memcpy(best.left_indices, left, n_left * sizeof(size_t));
                if (best.right_indices) memcpy(best.right_indices, right, n_right * sizeof(size_t));
            }
        }

        free(left);
        free(right);
        free(values);
    }

    return best;
}

static TreeNode* build_tree_classification(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int n_classes,
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int depth,
    int *n_nodes
) {
    TreeNode *node = calloc(1, sizeof(TreeNode));
    if (!node) return NULL;

    (*n_nodes)++;
    node->n_samples = n;
    node->n_classes = n_classes;

    double (*impurity_func)(const double*, const size_t*, size_t, int);
    impurity_func = (criterion == CRITERION_ENTROPY) ? entropy_impurity : gini_impurity;
    node->impurity = impurity_func(y, indices, n, n_classes);

    // Check stopping criteria
    int should_stop = (
        depth >= max_depth ||
        (int)n < min_samples_split ||
        node->impurity < 1e-10  // Pure node
    );

    if (should_stop) {
        node->is_leaf = 1;
        node->class_label = majority_class(y, indices, n, n_classes);
        node->class_proba = class_probabilities(y, indices, n, n_classes);
        return node;
    }

    // Find best split
    BestSplit best = find_best_split_classification(X, y, indices, n, n_classes, criterion);

    // Check if split is valid
    if (best.impurity_decrease < min_impurity_decrease ||
        (int)best.n_left < min_samples_leaf ||
        (int)best.n_right < min_samples_leaf) {
        node->is_leaf = 1;
        node->class_label = majority_class(y, indices, n, n_classes);
        node->class_proba = class_probabilities(y, indices, n, n_classes);
        free(best.left_indices);
        free(best.right_indices);
        return node;
    }

    // Build children
    node->is_leaf = 0;
    node->feature_index = best.feature_index;
    node->threshold = best.threshold;

    node->left = build_tree_classification(
        X, y, best.left_indices, best.n_left, n_classes,
        criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    node->right = build_tree_classification(
        X, y, best.right_indices, best.n_right, n_classes,
        criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    free(best.left_indices);
    free(best.right_indices);

    return node;
}

/* ============================================
 * Tree building (Regression)
 * ============================================ */

static BestSplit find_best_split_regression(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n
) {
    BestSplit best = {0, 0.0, -DBL_MAX, NULL, 0, NULL, 0};

    double parent_mse = mse_impurity(y, indices, n);

    for (size_t feat = 0; feat < X->cols; feat++) {
        double *values = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            values[i] = X->data[indices[i] * X->cols + feat];
        }

        // Sort values for threshold search
        qsort(values, n, sizeof(double), compare_doubles);

        // Pre-allocate split buffers outside the threshold loop
        size_t *left = malloc(n * sizeof(size_t));
        size_t *right = malloc(n * sizeof(size_t));
        if (!left || !right) {
            free(values);
            free(left);
            free(right);
            continue;
        }

        for (size_t t = 0; t < n - 1; t++) {
            if (values[t] == values[t + 1]) continue;

            double threshold = (values[t] + values[t + 1]) / 2.0;

            // Split indices (reuse pre-allocated buffers)
            size_t n_left = 0, n_right = 0;

            for (size_t i = 0; i < n; i++) {
                double val = X->data[indices[i] * X->cols + feat];
                if (val <= threshold) {
                    left[n_left++] = indices[i];
                } else {
                    right[n_right++] = indices[i];
                }
            }

            if (n_left == 0 || n_right == 0) {
                continue;
            }

            double left_mse = mse_impurity(y, left, n_left);
            double right_mse = mse_impurity(y, right, n_right);
            double weighted_mse = (n_left * left_mse + n_right * right_mse) / n;
            double impurity_decrease = parent_mse - weighted_mse;

            if (impurity_decrease > best.impurity_decrease) {
                free(best.left_indices);
                free(best.right_indices);

                best.feature_index = feat;
                best.threshold = threshold;
                best.impurity_decrease = impurity_decrease;
                best.n_left = n_left;
                best.n_right = n_right;

                // Allocate new buffers for best, reuse left/right for next iteration
                best.left_indices = malloc(n_left * sizeof(size_t));
                best.right_indices = malloc(n_right * sizeof(size_t));
                if (best.left_indices) memcpy(best.left_indices, left, n_left * sizeof(size_t));
                if (best.right_indices) memcpy(best.right_indices, right, n_right * sizeof(size_t));
            }
        }

        free(left);
        free(right);
        free(values);
    }

    return best;
}

static TreeNode* build_tree_regression(
    const Matrix *X,
    const double *y,
    const size_t *indices,
    size_t n,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int depth,
    int *n_nodes
) {
    TreeNode *node = calloc(1, sizeof(TreeNode));
    if (!node) return NULL;

    (*n_nodes)++;
    node->n_samples = n;
    node->impurity = mse_impurity(y, indices, n);

    int should_stop = (
        depth >= max_depth ||
        (int)n < min_samples_split ||
        node->impurity < 1e-10
    );

    if (should_stop) {
        node->is_leaf = 1;
        node->value = mean_value(y, indices, n);
        return node;
    }

    BestSplit best = find_best_split_regression(X, y, indices, n);

    if (best.impurity_decrease < min_impurity_decrease ||
        (int)best.n_left < min_samples_leaf ||
        (int)best.n_right < min_samples_leaf) {
        node->is_leaf = 1;
        node->value = mean_value(y, indices, n);
        free(best.left_indices);
        free(best.right_indices);
        return node;
    }

    node->is_leaf = 0;
    node->feature_index = best.feature_index;
    node->threshold = best.threshold;

    node->left = build_tree_regression(
        X, y, best.left_indices, best.n_left,
        max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    node->right = build_tree_regression(
        X, y, best.right_indices, best.n_right,
        max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, depth + 1, n_nodes
    );

    free(best.left_indices);
    free(best.right_indices);

    return node;
}

/* ============================================
 * Decision Tree Classifier API
 * ============================================ */

DecisionTreeClassifier* decision_tree_classifier_create(void) {
    return decision_tree_classifier_create_full(CRITERION_GINI, 10, 2, 1, 0.0);
}

DecisionTreeClassifier* decision_tree_classifier_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
) {
    DecisionTreeClassifier *tree = calloc(1, sizeof(DecisionTreeClassifier));
    if (!tree) return NULL;

    tree->base.type = MODEL_DECISION_TREE;
    tree->base.task = TASK_CLASSIFICATION;
    tree->base.is_fitted = 0;
    tree->base.verbose = VERBOSE_SILENT;

    tree->base.fit = decision_tree_classifier_fit;
    tree->base.predict = decision_tree_classifier_predict;
    tree->base.predict_proba = decision_tree_classifier_predict_proba;
    tree->base.transform = NULL;
    tree->base.score = decision_tree_classifier_score;
    tree->base.clone = decision_tree_classifier_clone;
    tree->base.free = decision_tree_classifier_free;
    tree->base.save = NULL;
    tree->base.load = NULL;
    tree->base.print_summary = NULL;

    tree->root = NULL;
    tree->criterion = criterion;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->min_impurity_decrease = min_impurity_decrease;
    tree->max_features = 0;

    return tree;
}

Estimator* decision_tree_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    DecisionTreeClassifier *tree = (DecisionTreeClassifier*)self;

    // Free previous tree
    if (tree->root) {
        tree_node_free(tree->root);
        tree->root = NULL;
    }

    // Count classes
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    tree->n_classes = max_class + 1;
    tree->n_features = X->cols;

    // Create indices
    size_t *indices = malloc(X->rows * sizeof(size_t));
    for (size_t i = 0; i < X->rows; i++) {
        indices[i] = i;
    }

    // Build tree
    tree->n_nodes_ = 0;
    tree->root = build_tree_classification(
        X, y->data, indices, X->rows, tree->n_classes,
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease, 0, &tree->n_nodes_
    );

    free(indices);
    tree->base.is_fitted = 1;

    return self;
}

static int predict_single_class(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->class_label;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_single_class(node->left, x);
    } else {
        return predict_single_class(node->right, x);
    }
}

Matrix* decision_tree_classifier_predict(const Estimator *self, const Matrix *X) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_classifier_predict: model not fitted");
        return NULL;
    }

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] = predict_single_class(tree->root, &X->data[i * X->cols]);
    }

    return predictions;
}

static const double* predict_proba_single(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->class_proba;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_proba_single(node->left, x);
    } else {
        return predict_proba_single(node->right, x);
    }
}

Matrix* decision_tree_classifier_predict_proba(const Estimator *self, const Matrix *X) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_classifier_predict_proba: model not fitted");
        return NULL;
    }

    Matrix *proba = matrix_alloc(X->rows, tree->n_classes);
    if (!proba) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        const double *p = predict_proba_single(tree->root, &X->data[i * X->cols]);
        for (int c = 0; c < tree->n_classes; c++) {
            proba->data[i * tree->n_classes + c] = p ? p[c] : 0.0;
        }
    }

    return proba;
}

double decision_tree_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

Estimator* decision_tree_classifier_clone(const Estimator *self) {
    const DecisionTreeClassifier *tree = (const DecisionTreeClassifier*)self;
    return (Estimator*)decision_tree_classifier_create_full(
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease
    );
}

void tree_node_free(TreeNode *node) {
    if (!node) return;
    tree_node_free(node->left);
    tree_node_free(node->right);
    free(node->class_proba);
    free(node);
}

void decision_tree_classifier_free(Estimator *self) {
    DecisionTreeClassifier *tree = (DecisionTreeClassifier*)self;
    if (tree) {
        tree_node_free(tree->root);
        free(tree);
    }
}

void decision_tree_export_text(const TreeNode *root, int depth) {
    if (!root) return;

    for (int i = 0; i < depth; i++) printf("  ");

    if (root->is_leaf) {
        printf("return class %d (samples=%zu)\n", root->class_label, root->n_samples);
    } else {
        printf("if feature[%zu] <= %.4f:\n", root->feature_index, root->threshold);
        decision_tree_export_text(root->left, depth + 1);

        for (int i = 0; i < depth; i++) printf("  ");
        printf("else:\n");
        decision_tree_export_text(root->right, depth + 1);
    }
}

/* ============================================
 * Decision Tree Regressor API
 * ============================================ */

DecisionTreeRegressor* decision_tree_regressor_create(void) {
    return decision_tree_regressor_create_full(CRITERION_MSE, 10, 2, 1, 0.0);
}

DecisionTreeRegressor* decision_tree_regressor_create_full(
    SplitCriterion criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease
) {
    DecisionTreeRegressor *tree = calloc(1, sizeof(DecisionTreeRegressor));
    if (!tree) return NULL;

    tree->base.type = MODEL_DECISION_TREE;
    tree->base.task = TASK_REGRESSION;
    tree->base.is_fitted = 0;
    tree->base.verbose = VERBOSE_SILENT;

    tree->base.fit = decision_tree_regressor_fit;
    tree->base.predict = decision_tree_regressor_predict;
    tree->base.predict_proba = NULL;
    tree->base.transform = NULL;
    tree->base.score = decision_tree_regressor_score;
    tree->base.clone = decision_tree_regressor_clone;
    tree->base.free = decision_tree_regressor_free;
    tree->base.save = NULL;
    tree->base.load = NULL;
    tree->base.print_summary = NULL;

    tree->root = NULL;
    tree->criterion = criterion;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->min_impurity_decrease = min_impurity_decrease;
    tree->max_features = 0;

    return tree;
}

Estimator* decision_tree_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    DecisionTreeRegressor *tree = (DecisionTreeRegressor*)self;

    if (tree->root) {
        tree_node_free(tree->root);
        tree->root = NULL;
    }

    tree->n_features = X->cols;

    size_t *indices = malloc(X->rows * sizeof(size_t));
    for (size_t i = 0; i < X->rows; i++) {
        indices[i] = i;
    }

    tree->n_nodes_ = 0;
    tree->root = build_tree_regression(
        X, y->data, indices, X->rows,
        tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease, 0, &tree->n_nodes_
    );

    free(indices);
    tree->base.is_fitted = 1;

    return self;
}

static double predict_single_value(const TreeNode *node, const double *x) {
    if (node->is_leaf) {
        return node->value;
    }

    if (x[node->feature_index] <= node->threshold) {
        return predict_single_value(node->left, x);
    } else {
        return predict_single_value(node->right, x);
    }
}

Matrix* decision_tree_regressor_predict(const Estimator *self, const Matrix *X) {
    const DecisionTreeRegressor *tree = (const DecisionTreeRegressor*)self;

    if (!tree->base.is_fitted || !tree->root) {
        cml_set_error(CML_ERROR_NOT_FITTED, "decision_tree_regressor_predict: model not fitted");
        return NULL;
    }

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] = predict_single_value(tree->root, &X->data[i * X->cols]);
    }

    return predictions;
}

double decision_tree_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* decision_tree_regressor_clone(const Estimator *self) {
    const DecisionTreeRegressor *tree = (const DecisionTreeRegressor*)self;
    return (Estimator*)decision_tree_regressor_create_full(
        tree->criterion, tree->max_depth, tree->min_samples_split,
        tree->min_samples_leaf, tree->min_impurity_decrease
    );
}

void decision_tree_regressor_free(Estimator *self) {
    DecisionTreeRegressor *tree = (DecisionTreeRegressor*)self;
    if (tree) {
        tree_node_free(tree->root);
        free(tree);
    }
}


/* --- decomposition.c --- */
/**
 * decomposition.c - PCA implementation using power iteration
 *
 * Uses iterative method for eigenvector computation - pure C, no LAPACK dependency
 */





#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Helper Functions
 * ============================================ */

Matrix* compute_covariance_matrix(const Matrix *X, const Matrix *mean) {
    size_t n = X->rows;
    size_t m = X->cols;

    // Center data
    Matrix *X_centered = matrix_alloc(n, m);
    if (!X_centered) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            X_centered->data[i * m + j] = X->data[i * m + j] - mean->data[j];
        }
    }

    // Cov = X_centered' @ X_centered / (n - 1)
    Matrix *X_t = matrix_transpose(X_centered);
    Matrix *cov = matrix_matmul(X_t, X_centered);

    if (cov) {
        double scale = (n > 1) ? 1.0 / (n - 1) : 1.0;
        for (size_t i = 0; i < cov->rows * cov->cols; i++) {
            cov->data[i] *= scale;
        }
    }

    matrix_free(X_centered);
    matrix_free(X_t);

    return cov;
}

Matrix* power_iteration(const Matrix *A, int max_iter, double tol) {
    size_t n = A->rows;

    // Initialize random vector
    Matrix *v = matrix_alloc(n, 1);
    if (!v) return NULL;

    for (size_t i = 0; i < n; i++) {
        v->data[i] = rand_uniform() - 0.5;
    }

    // Normalize
    double norm = vector_norm(v);
    for (size_t i = 0; i < n; i++) {
        v->data[i] /= norm;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // v_new = A @ v
        Matrix *v_new = matrix_matmul(A, v);
        if (!v_new) {
            matrix_free(v);
            return NULL;
        }

        // Normalize
        norm = vector_norm(v_new);
        if (norm < 1e-10) {
            matrix_free(v_new);
            break;
        }
        for (size_t i = 0; i < n; i++) {
            v_new->data[i] /= norm;
        }

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = fabs(v_new->data[i]) - fabs(v->data[i]);
            diff += d * d;
        }
        diff = sqrt(diff);

        matrix_free(v);
        v = v_new;

        if (diff < tol) break;
    }

    return v;
}

// Deflation: A = A - eigenvalue * v @ v.T
static void deflate_matrix(Matrix *A, const Matrix *v, double eigenvalue) {
    size_t n = A->rows;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A->data[i * n + j] -= eigenvalue * v->data[i] * v->data[j];
        }
    }
}

int eigen_decomposition(const Matrix *A, Matrix **eigenvectors, double *eigenvalues, int n_components) {
    size_t n = A->rows;

    // Create copy of A for deflation
    Matrix *A_work = matrix_copy(A);
    if (!A_work) return -1;

    *eigenvectors = matrix_alloc(n_components, n);
    if (!*eigenvectors) {
        matrix_free(A_work);
        return -1;
    }

    for (int k = 0; k < n_components; k++) {
        // Find dominant eigenvector
        Matrix *v = power_iteration(A_work, 1000, 1e-8);
        if (!v) {
            matrix_free(A_work);
            matrix_free(*eigenvectors);
            return -1;
        }

        // Compute eigenvalue: lambda = v.T @ A @ v
        Matrix *Av = matrix_matmul(A_work, v);
        double eigenvalue = 0.0;
        for (size_t i = 0; i < n; i++) {
            eigenvalue += v->data[i] * Av->data[i];
        }
        matrix_free(Av);

        eigenvalues[k] = eigenvalue;

        // Store eigenvector as row
        for (size_t i = 0; i < n; i++) {
            (*eigenvectors)->data[k * n + i] = v->data[i];
        }

        // Deflate matrix
        deflate_matrix(A_work, v, eigenvalue);

        matrix_free(v);
    }

    matrix_free(A_work);
    return 0;
}

/* ============================================
 * PCA Implementation
 * ============================================ */

PCA* pca_create(int n_components) {
    return pca_create_full(n_components, 0);
}

PCA* pca_create_full(int n_components, int whiten) {
    PCA *pca = calloc(1, sizeof(PCA));
    if (!pca) return NULL;

    pca->base.type = MODEL_PCA;
    pca->base.task = TASK_TRANSFORMATION;
    pca->base.is_fitted = 0;
    pca->base.verbose = VERBOSE_SILENT;

    pca->base.fit = pca_fit;
    pca->base.predict = NULL;
    pca->base.predict_proba = NULL;
    pca->base.transform = pca_transform;
    pca->base.score = NULL;
    pca->base.clone = pca_clone;
    pca->base.free = pca_free;
    pca->base.save = NULL;
    pca->base.load = NULL;
    pca->base.print_summary = pca_print_summary;

    pca->n_components = n_components;
    pca->whiten = whiten;
    pca->components_ = NULL;
    pca->mean_ = NULL;
    pca->explained_variance_ = NULL;
    pca->explained_variance_ratio_ = NULL;
    pca->singular_values_ = NULL;

    return pca;
}

Estimator* pca_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;  // PCA doesn't use y
    PCA *pca = (PCA*)self;

    pca->n_samples_ = X->rows;
    pca->n_features_ = X->cols;

    // Determine number of components
    int n_comp = pca->n_components;
    if (n_comp <= 0 || n_comp > (int)X->cols) {
        n_comp = X->cols;
    }
    pca->n_components = n_comp;

    // Compute mean
    pca->mean_ = matrix_alloc(1, X->cols);
    for (size_t j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            sum += X->data[i * X->cols + j];
        }
        pca->mean_->data[j] = sum / X->rows;
    }

    // Compute covariance matrix
    Matrix *cov = compute_covariance_matrix(X, pca->mean_);
    if (!cov) return NULL;

    // Compute total variance
    pca->total_variance_ = 0.0;
    for (size_t i = 0; i < cov->rows; i++) {
        pca->total_variance_ += cov->data[i * cov->cols + i];
    }

    // Eigendecomposition
    pca->explained_variance_ = malloc(n_comp * sizeof(double));
    if (eigen_decomposition(cov, &pca->components_, pca->explained_variance_, n_comp) != 0) {
        matrix_free(cov);
        return NULL;
    }

    matrix_free(cov);

    // Compute explained variance ratio
    pca->explained_variance_ratio_ = malloc(n_comp * sizeof(double));
    pca->singular_values_ = malloc(n_comp * sizeof(double));

    for (int k = 0; k < n_comp; k++) {
        pca->explained_variance_ratio_[k] = pca->explained_variance_[k] / pca->total_variance_;
        pca->singular_values_[k] = sqrt(pca->explained_variance_[k] * (X->rows - 1));
    }

    pca->base.is_fitted = 1;

    if (pca->base.verbose >= VERBOSE_MINIMAL) {
        printf("PCA fitted: %d components, %.2f%% variance explained\n",
               n_comp, pca_cumulative_variance(pca, n_comp) * 100);
    }

    return self;
}

Matrix* pca_transform(const Estimator *self, const Matrix *X) {
    const PCA *pca = (const PCA*)self;

    if (!pca->base.is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "PCA not fitted");
        return NULL;
    }

    size_t n = X->rows;
    size_t m = X->cols;
    int k = pca->n_components;

    // Center data
    Matrix *X_centered = matrix_alloc(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            X_centered->data[i * m + j] = X->data[i * m + j] - pca->mean_->data[j];
        }
    }

    // Project: X_reduced = X_centered @ components.T
    Matrix *components_t = matrix_transpose(pca->components_);
    Matrix *X_reduced = matrix_matmul(X_centered, components_t);

    matrix_free(X_centered);
    matrix_free(components_t);

    // Whiten if requested
    if (pca->whiten && X_reduced) {
        for (size_t i = 0; i < X_reduced->rows; i++) {
            for (int j = 0; j < k; j++) {
                if (pca->singular_values_[j] > 1e-10) {
                    X_reduced->data[i * k + j] /= pca->singular_values_[j];
                }
            }
        }
    }

    return X_reduced;
}

Matrix* pca_fit_transform(Estimator *self, const Matrix *X, const Matrix *y) {
    if (!pca_fit(self, X, y)) return NULL;
    return pca_transform(self, X);
}

Matrix* pca_inverse_transform(const PCA *pca, const Matrix *X_reduced) {
    if (!pca->base.is_fitted) return NULL;

    // X_original = X_reduced @ components + mean
    Matrix *X_reconstructed = matrix_matmul(X_reduced, pca->components_);
    if (!X_reconstructed) return NULL;

    // Add mean back
    for (size_t i = 0; i < X_reconstructed->rows; i++) {
        for (size_t j = 0; j < X_reconstructed->cols; j++) {
            X_reconstructed->data[i * X_reconstructed->cols + j] += pca->mean_->data[j];
        }
    }

    return X_reconstructed;
}

const double* pca_explained_variance_ratio(const PCA *pca) {
    return pca ? pca->explained_variance_ratio_ : NULL;
}

double pca_cumulative_variance(const PCA *pca, int n_components) {
    if (!pca || !pca->explained_variance_ratio_) return 0.0;

    double sum = 0.0;
    int n = n_components;
    if (n > pca->n_components) n = pca->n_components;

    for (int i = 0; i < n; i++) {
        sum += pca->explained_variance_ratio_[i];
    }
    return sum;
}

Estimator* pca_clone(const Estimator *self) {
    const PCA *pca = (const PCA*)self;
    return (Estimator*)pca_create_full(pca->n_components, pca->whiten);
}

void pca_free(Estimator *self) {
    PCA *pca = (PCA*)self;
    if (!pca) return;

    matrix_free(pca->components_);
    matrix_free(pca->mean_);
    free(pca->explained_variance_);
    free(pca->explained_variance_ratio_);
    free(pca->singular_values_);
    free(pca);
}

void pca_print_summary(const Estimator *self) {
    const PCA *pca = (const PCA*)self;

    printf("\n=== PCA Summary ===\n");
    printf("Fitted: %s\n", pca->base.is_fitted ? "Yes" : "No");
    printf("Components: %d\n", pca->n_components);
    printf("Whiten: %s\n", pca->whiten ? "Yes" : "No");

    if (pca->base.is_fitted) {
        printf("\nExplained variance ratio:\n");
        double cumulative = 0.0;
        for (int i = 0; i < pca->n_components && i < 10; i++) {
            cumulative += pca->explained_variance_ratio_[i];
            printf("  PC%d: %.4f (cumulative: %.4f)\n",
                   i + 1, pca->explained_variance_ratio_[i], cumulative);
        }
        if (pca->n_components > 10) {
            printf("  ... (%d more)\n", pca->n_components - 10);
        }
        printf("\nTotal variance explained: %.4f\n", pca_cumulative_variance(pca, pca->n_components));
    }
    printf("===================\n\n");
}


/* --- neural_network.c --- */
/**
 * neural_network.c - Neural Network implementation
 *
 * Implements feedforward neural network with backpropagation
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Activation Functions
 * ============================================ */

static double nn_sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

Matrix* activation_forward(const Matrix *z, ActivationType type) {
    Matrix *a = matrix_alloc(z->rows, z->cols);
    if (!a) return NULL;

    switch (type) {
        case ACTIVATION_IDENTITY:
            memcpy(a->data, z->data, z->rows * z->cols * sizeof(double));
            break;

        case ACTIVATION_SIGMOID:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = nn_sigmoid(z->data[i]);
            }
            break;

        case ACTIVATION_TANH:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = tanh(z->data[i]);
            }
            break;

        case ACTIVATION_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = z->data[i] > 0 ? z->data[i] : 0;
            }
            break;

        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = z->data[i] > 0 ? z->data[i] : 0.01 * z->data[i];
            }
            break;

        case ACTIVATION_SOFTMAX:
            // Softmax per row
            for (size_t i = 0; i < z->rows; i++) {
                // Find max for numerical stability
                double max_val = z->data[i * z->cols];
                for (size_t j = 1; j < z->cols; j++) {
                    if (z->data[i * z->cols + j] > max_val) {
                        max_val = z->data[i * z->cols + j];
                    }
                }

                // Compute exp and sum
                double sum = 0.0;
                for (size_t j = 0; j < z->cols; j++) {
                    a->data[i * z->cols + j] = exp(z->data[i * z->cols + j] - max_val);
                    sum += a->data[i * z->cols + j];
                }

                // Normalize
                for (size_t j = 0; j < z->cols; j++) {
                    a->data[i * z->cols + j] /= sum;
                }
            }
            break;
    }

    return a;
}

Matrix* activation_derivative(const Matrix *z, const Matrix *a, ActivationType type) {
    Matrix *d = matrix_alloc(z->rows, z->cols);
    if (!d) return NULL;

    switch (type) {
        case ACTIVATION_IDENTITY:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0;
            }
            break;

        case ACTIVATION_SIGMOID:
            // nn_sigmoid'(x) = nn_sigmoid(x) * (1 - nn_sigmoid(x))
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = a->data[i] * (1.0 - a->data[i]);
            }
            break;

        case ACTIVATION_TANH:
            // tanh'(x) = 1 - tanh²(x)
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0 - a->data[i] * a->data[i];
            }
            break;

        case ACTIVATION_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = z->data[i] > 0 ? 1.0 : 0.0;
            }
            break;

        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = z->data[i] > 0 ? 1.0 : 0.01;
            }
            break;

        case ACTIVATION_SOFTMAX:
            // For softmax with cross-entropy, derivative is handled specially
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0;  // Will be handled in loss
            }
            break;
    }

    return d;
}

/* ============================================
 * Layer Management
 * ============================================ */

Layer* layer_create(int n_input, int n_output, ActivationType activation) {
    Layer *layer = calloc(1, sizeof(Layer));
    if (!layer) return NULL;

    layer->weights = matrix_alloc(n_input, n_output);
    layer->biases = matrix_alloc(1, n_output);
    layer->d_weights = matrix_alloc(n_input, n_output);
    layer->d_biases = matrix_alloc(1, n_output);
    layer->activation = activation;

    if (!layer->weights || !layer->biases || !layer->d_weights || !layer->d_biases) {
        layer_free(layer);
        return NULL;
    }

    // Xavier initialization
    double std = sqrt(2.0 / (n_input + n_output));
    for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] = rand_normal() * std;
    }

    // Initialize biases to small values
    for (size_t i = 0; i < layer->biases->cols; i++) {
        layer->biases->data[i] = 0.01;
    }

    return layer;
}

void layer_free(Layer *layer) {
    if (!layer) return;
    matrix_free(layer->weights);
    matrix_free(layer->biases);
    matrix_free(layer->d_weights);
    matrix_free(layer->d_biases);
    matrix_free(layer->v_weights);
    matrix_free(layer->v_biases);
    matrix_free(layer->m_weights);
    matrix_free(layer->m_biases);
    matrix_free(layer->input_cache);
    matrix_free(layer->output_cache);
    matrix_free(layer->z_cache);
    free(layer);
}

/* ============================================
 * Neural Network Implementation
 * ============================================ */

NeuralNetwork* neural_network_create(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType activation
) {
    return neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        activation, ACTIVATION_IDENTITY,  // Output will be set in init
        LOSS_MSE, OPTIMIZER_SGD,
        0.001, 200, 32, 0.9, 0.0001
    );
}

NeuralNetwork* neural_network_create_full(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType hidden_activation,
    ActivationType output_activation,
    LossType loss,
    OptimizerType optimizer,
    double learning_rate,
    int max_epochs,
    int batch_size,
    double momentum,
    double l2_reg
) {
    NeuralNetwork *nn = calloc(1, sizeof(NeuralNetwork));
    if (!nn) return NULL;

    // Store hidden layer sizes
    nn->layer_sizes = malloc((n_hidden + 2) * sizeof(int));  // +2 for input/output
    if (!nn->layer_sizes) {
        free(nn);
        return NULL;
    }
    for (int i = 0; i < n_hidden; i++) {
        nn->layer_sizes[i + 1] = hidden_layer_sizes[i];
    }
    nn->n_layer_sizes = n_hidden + 2;

    // Setup Estimator base
    nn->base.type = MODEL_NEURAL_NETWORK;
    nn->base.task = TASK_CLASSIFICATION;  // Will be set based on output
    nn->base.is_fitted = 0;
    nn->base.verbose = VERBOSE_SILENT;

    nn->base.fit = neural_network_fit;
    nn->base.predict = neural_network_predict;
    nn->base.predict_proba = neural_network_predict_proba;
    nn->base.transform = NULL;
    nn->base.score = neural_network_score;
    nn->base.clone = neural_network_clone;
    nn->base.free = neural_network_free;
    nn->base.save = NULL;
    nn->base.load = NULL;
    nn->base.print_summary = neural_network_print_summary;

    // Hyperparameters
    nn->learning_rate = learning_rate;
    nn->max_epochs = max_epochs;
    nn->batch_size = batch_size;
    nn->momentum = momentum;
    nn->beta1 = 0.9;
    nn->beta2 = 0.999;
    nn->epsilon = 1e-8;
    nn->l2_reg = l2_reg;
    nn->dropout_rate = 0.0;

    nn->loss_type = loss;
    nn->optimizer = optimizer;
    nn->hidden_activation = hidden_activation;
    nn->output_activation = output_activation;

    nn->layers = NULL;
    nn->n_layers = 0;
    nn->epoch = 0;
    nn->t = 0;

    return nn;
}

void neural_network_init(NeuralNetwork *nn, int n_features, int n_outputs) {
    if (!nn) return;

    nn->n_features = n_features;
    nn->n_classes = n_outputs;

    // Set layer sizes
    nn->layer_sizes[0] = n_features;
    nn->layer_sizes[nn->n_layer_sizes - 1] = n_outputs;

    // Create layers
    nn->n_layers = nn->n_layer_sizes - 1;
    nn->layers = malloc(nn->n_layers * sizeof(Layer*));
    if (!nn->layers) return;

    /* RNG seeded by caller or auto-seeded on first use */

    for (int i = 0; i < nn->n_layers; i++) {
        ActivationType act;
        if (i == nn->n_layers - 1) {
            // Output layer
            act = nn->output_activation;
        } else {
            // Hidden layers
            act = nn->hidden_activation;
        }

        nn->layers[i] = layer_create(nn->layer_sizes[i], nn->layer_sizes[i + 1], act);
        if (!nn->layers[i]) return;

        // Initialize momentum/Adam if needed
        if (nn->optimizer == OPTIMIZER_MOMENTUM || nn->optimizer == OPTIMIZER_ADAM) {
            nn->layers[i]->v_weights = matrix_alloc(nn->layer_sizes[i], nn->layer_sizes[i + 1]);
            nn->layers[i]->v_biases = matrix_alloc(1, nn->layer_sizes[i + 1]);
            if (!nn->layers[i]->v_weights || !nn->layers[i]->v_biases) { return; }
            matrix_fill(nn->layers[i]->v_weights, 0);
            matrix_fill(nn->layers[i]->v_biases, 0);
        }
        if (nn->optimizer == OPTIMIZER_ADAM) {
            nn->layers[i]->m_weights = matrix_alloc(nn->layer_sizes[i], nn->layer_sizes[i + 1]);
            nn->layers[i]->m_biases = matrix_alloc(1, nn->layer_sizes[i + 1]);
            if (!nn->layers[i]->m_weights || !nn->layers[i]->m_biases) { return; }
            matrix_fill(nn->layers[i]->m_weights, 0);
            matrix_fill(nn->layers[i]->m_biases, 0);
        }
    }
}

Matrix* neural_network_forward(NeuralNetwork *nn, const Matrix *X) {
    Matrix *current = matrix_copy(X);
    if (!current) return NULL;

    for (int i = 0; i < nn->n_layers; i++) {
        Layer *layer = nn->layers[i];

        // Store input for backprop
        matrix_free(layer->input_cache);
        layer->input_cache = matrix_copy(current);

        // z = X @ W + b
        Matrix *z = matrix_matmul(current, layer->weights);
        if (!z) {
            matrix_free(current);
            return NULL;
        }

        // Add bias (broadcast)
        for (size_t r = 0; r < z->rows; r++) {
            for (size_t c = 0; c < z->cols; c++) {
                z->data[r * z->cols + c] += layer->biases->data[c];
            }
        }

        // Store z for backprop
        matrix_free(layer->z_cache);
        layer->z_cache = matrix_copy(z);

        // Apply activation
        Matrix *a = activation_forward(z, layer->activation);
        matrix_free(z);
        matrix_free(current);

        if (!a) return NULL;

        // Store output
        matrix_free(layer->output_cache);
        layer->output_cache = matrix_copy(a);

        current = a;
    }

    return current;
}

void neural_network_backward(NeuralNetwork *nn, const Matrix *y) {
    // Get output layer
    Layer *output_layer = nn->layers[nn->n_layers - 1];
    Matrix *output = output_layer->output_cache;

    // Compute output error (delta)
    Matrix *delta = matrix_alloc(output->rows, output->cols);

    // For softmax + cross-entropy or nn_sigmoid + BCE: delta = output - y
    for (size_t i = 0; i < output->rows * output->cols; i++) {
        delta->data[i] = output->data[i] - y->data[i];
    }

    // Backpropagate through layers
    for (int i = nn->n_layers - 1; i >= 0; i--) {
        Layer *layer = nn->layers[i];
        size_t batch_size = layer->input_cache->rows;

        // Compute gradients
        // d_weights = input.T @ delta / batch_size
        Matrix *input_t = matrix_transpose(layer->input_cache);
        Matrix *d_w = matrix_matmul(input_t, delta);
        if (!d_w) { matrix_free(input_t); matrix_free(delta); return; }
        matrix_free(input_t);

        for (size_t j = 0; j < d_w->rows * d_w->cols; j++) {
            layer->d_weights->data[j] = d_w->data[j] / batch_size;
            // Add L2 regularization
            layer->d_weights->data[j] += nn->l2_reg * layer->weights->data[j];
        }
        matrix_free(d_w);

        // d_biases = mean(delta, axis=0)
        for (size_t j = 0; j < layer->biases->cols; j++) {
            double sum = 0;
            for (size_t r = 0; r < batch_size; r++) {
                sum += delta->data[r * delta->cols + j];
            }
            layer->d_biases->data[j] = sum / batch_size;
        }

        // Propagate delta to previous layer (if not input layer)
        if (i > 0) {
            Layer *prev_layer = nn->layers[i - 1];

            // delta_prev = delta @ W.T * activation'(z)
            Matrix *w_t = matrix_transpose(layer->weights);
            Matrix *delta_back = matrix_matmul(delta, w_t);
            if (!delta_back) { matrix_free(w_t); matrix_free(delta); return; }
            matrix_free(w_t);
            matrix_free(delta);

            // Multiply by activation derivative
            Matrix *act_deriv = activation_derivative(prev_layer->z_cache, prev_layer->output_cache, prev_layer->activation);
            for (size_t j = 0; j < delta_back->rows * delta_back->cols; j++) {
                delta_back->data[j] *= act_deriv->data[j];
            }
            matrix_free(act_deriv);

            delta = delta_back;
        }
    }

    matrix_free(delta);
}

void neural_network_update_weights(NeuralNetwork *nn) {
    nn->t++;  // Increment time step

    for (int i = 0; i < nn->n_layers; i++) {
        Layer *layer = nn->layers[i];
        size_t n_weights = layer->weights->rows * layer->weights->cols;
        size_t n_biases = layer->biases->cols;

        if (nn->optimizer == OPTIMIZER_SGD) {
            // Standard SGD
            for (size_t j = 0; j < n_weights; j++) {
                layer->weights->data[j] -= nn->learning_rate * layer->d_weights->data[j];
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->biases->data[j] -= nn->learning_rate * layer->d_biases->data[j];
            }
        } else if (nn->optimizer == OPTIMIZER_MOMENTUM) {
            // SGD with momentum
            for (size_t j = 0; j < n_weights; j++) {
                layer->v_weights->data[j] = nn->momentum * layer->v_weights->data[j] - nn->learning_rate * layer->d_weights->data[j];
                layer->weights->data[j] += layer->v_weights->data[j];
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->v_biases->data[j] = nn->momentum * layer->v_biases->data[j] - nn->learning_rate * layer->d_biases->data[j];
                layer->biases->data[j] += layer->v_biases->data[j];
            }
        } else if (nn->optimizer == OPTIMIZER_ADAM) {
            // Adam optimizer
            double bias_correction1 = 1.0 - pow(nn->beta1, nn->t);
            double bias_correction2 = 1.0 - pow(nn->beta2, nn->t);

            for (size_t j = 0; j < n_weights; j++) {
                layer->m_weights->data[j] = nn->beta1 * layer->m_weights->data[j] + (1 - nn->beta1) * layer->d_weights->data[j];
                layer->v_weights->data[j] = nn->beta2 * layer->v_weights->data[j] + (1 - nn->beta2) * layer->d_weights->data[j] * layer->d_weights->data[j];

                double m_hat = layer->m_weights->data[j] / bias_correction1;
                double v_hat = layer->v_weights->data[j] / bias_correction2;

                layer->weights->data[j] -= nn->learning_rate * m_hat / (sqrt(v_hat) + nn->epsilon);
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->m_biases->data[j] = nn->beta1 * layer->m_biases->data[j] + (1 - nn->beta1) * layer->d_biases->data[j];
                layer->v_biases->data[j] = nn->beta2 * layer->v_biases->data[j] + (1 - nn->beta2) * layer->d_biases->data[j] * layer->d_biases->data[j];

                double m_hat = layer->m_biases->data[j] / bias_correction1;
                double v_hat = layer->v_biases->data[j] / bias_correction2;

                layer->biases->data[j] -= nn->learning_rate * m_hat / (sqrt(v_hat) + nn->epsilon);
            }
        }
    }
}

Estimator* neural_network_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    NeuralNetwork *nn = (NeuralNetwork*)self;

    // Determine output dimension and type
    int n_outputs = y->cols;
    if (n_outputs == 1) {
        // Check if classification or regression
        int is_classification = 1;
        for (size_t i = 0; i < y->rows; i++) {
            double val = y->data[i];
            if (val != floor(val) || val < 0) {
                is_classification = 0;
                break;
            }
        }

        if (is_classification) {
            // Find max class and one-hot encode
            int max_class = 0;
            for (size_t i = 0; i < y->rows; i++) {
                int label = (int)y->data[i];
                if (label > max_class) max_class = label;
            }
            n_outputs = max_class + 1;
            nn->base.task = TASK_CLASSIFICATION;
            nn->output_activation = (n_outputs == 2) ? ACTIVATION_SIGMOID : ACTIVATION_SOFTMAX;
            nn->loss_type = (n_outputs == 2) ? LOSS_BINARY_CE : LOSS_CROSS_ENTROPY;
        } else {
            nn->base.task = TASK_REGRESSION;
            nn->output_activation = ACTIVATION_IDENTITY;
            nn->loss_type = LOSS_MSE;
        }
    }

    // Initialize network if not done
    if (!nn->layers) {
        neural_network_init(nn, X->cols, n_outputs);
    }

    // Prepare target (one-hot encoding for classification)
    Matrix *y_train = NULL;
    if (nn->base.task == TASK_CLASSIFICATION && y->cols == 1) {
        y_train = matrix_alloc(y->rows, n_outputs);
        for (size_t i = 0; i < y->rows; i++) {
            int label = (int)y->data[i];
            for (int c = 0; c < n_outputs; c++) {
                y_train->data[i * n_outputs + c] = (c == label) ? 1.0 : 0.0;
            }
        }
    } else {
        y_train = matrix_copy(y);
    }

    // Training history
    if (nn->base.verbose != VERBOSE_SILENT) {
        if (nn->base.history) training_history_free(nn->base.history);
        nn->base.history = training_history_alloc(nn->max_epochs);
    }

    // Training loop
    clock_t start = clock();
    size_t n_samples = X->rows;
    size_t n_batches = (n_samples + nn->batch_size - 1) / nn->batch_size;

    for (nn->epoch = 0; nn->epoch < nn->max_epochs; nn->epoch++) {
        double epoch_loss = 0.0;

        // Shuffle indices
        size_t *indices = malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) indices[i] = i;
        shuffle_indices(indices, n_samples);

        // Mini-batch training
        for (size_t batch = 0; batch < n_batches; batch++) {
            size_t batch_start = batch * nn->batch_size;
            size_t batch_end = batch_start + nn->batch_size;
            if (batch_end > n_samples) batch_end = n_samples;
            size_t actual_batch_size = batch_end - batch_start;

            // Get batch
            Matrix *X_batch = matrix_alloc(actual_batch_size, X->cols);
            Matrix *y_batch = matrix_alloc(actual_batch_size, y_train->cols);

            for (size_t i = 0; i < actual_batch_size; i++) {
                size_t idx = indices[batch_start + i];
                for (size_t j = 0; j < X->cols; j++) {
                    X_batch->data[i * X->cols + j] = X->data[idx * X->cols + j];
                }
                for (size_t j = 0; j < y_train->cols; j++) {
                    y_batch->data[i * y_train->cols + j] = y_train->data[idx * y_train->cols + j];
                }
            }

            // Forward
            Matrix *output = neural_network_forward(nn, X_batch);

            // Compute loss
            double batch_loss = 0.0;
            if (nn->loss_type == LOSS_MSE) {
                for (size_t i = 0; i < output->rows * output->cols; i++) {
                    double diff = output->data[i] - y_batch->data[i];
                    batch_loss += diff * diff;
                }
                batch_loss /= (output->rows * output->cols);
            } else {
                // Cross-entropy
                for (size_t i = 0; i < output->rows * output->cols; i++) {
                    double p = output->data[i];
                    double t = y_batch->data[i];
                    if (p > 1e-10 && p < 1 - 1e-10) {
                        batch_loss -= t * log(p) + (1 - t) * log(1 - p);
                    }
                }
                batch_loss /= output->rows;
            }
            epoch_loss += batch_loss;

            // Backward
            neural_network_backward(nn, y_batch);

            // Update weights
            neural_network_update_weights(nn);

            matrix_free(X_batch);
            matrix_free(y_batch);
            matrix_free(output);
        }

        free(indices);
        epoch_loss /= n_batches;

        // Compute accuracy/R² for history
        double metric = 0.0;
        if (nn->base.task == TASK_CLASSIFICATION) {
            Matrix *pred = neural_network_predict(self, X);
            metric = accuracy(y, pred);
            matrix_free(pred);
        } else {
            metric = 1.0 - epoch_loss;  // Approximate R²
        }

        // Record history
        if (nn->base.history) {
            training_history_append(nn->base.history, epoch_loss, metric);
        }

        // Verbose output
        const char *metric_name = nn->base.task == TASK_CLASSIFICATION ? "accuracy" : "R²";
        verbose_print_epoch(nn->base.verbose, nn->epoch + 1, nn->max_epochs, epoch_loss, metric, metric_name);

        // Callback
        if (nn->base.callback) {
            nn->base.callback(nn->epoch + 1, epoch_loss, metric, nn->base.callback_data);
        }
    }

    matrix_free(y_train);
    nn->base.is_fitted = 1;

    // Final verbose output
    if (nn->base.verbose >= VERBOSE_MINIMAL) {
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double final_score = neural_network_score(self, X, y);
        const char *metric_name = nn->base.task == TASK_CLASSIFICATION ? "accuracy" : "R²";
        verbose_print_final(nn->base.verbose, "NeuralNetwork", final_score, metric_name, elapsed);
    }

    return self;
}

Matrix* neural_network_predict(const Estimator *self, const Matrix *X) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn->base.is_fitted) return NULL;

    Matrix *output = neural_network_forward(nn, X);
    if (!output) return NULL;

    if (nn->base.task == TASK_CLASSIFICATION) {
        // Convert to class labels
        Matrix *predictions = matrix_alloc(output->rows, 1);
        for (size_t i = 0; i < output->rows; i++) {
            if (output->cols == 1) {
                // Binary classification
                predictions->data[i] = output->data[i] > 0.5 ? 1.0 : 0.0;
            } else {
                // Multi-class: argmax
                int max_class = 0;
                double max_val = output->data[i * output->cols];
                for (size_t c = 1; c < output->cols; c++) {
                    if (output->data[i * output->cols + c] > max_val) {
                        max_val = output->data[i * output->cols + c];
                        max_class = c;
                    }
                }
                predictions->data[i] = max_class;
            }
        }
        matrix_free(output);
        return predictions;
    }

    return output;
}

Matrix* neural_network_predict_proba(const Estimator *self, const Matrix *X) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn->base.is_fitted) return NULL;
    return neural_network_forward(nn, X);
}

double neural_network_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;
    if (nn->base.task == TASK_CLASSIFICATION) {
        return classification_score_accuracy(self, X, y);
    }
    return regression_score_r2(self, X, y);
}

Estimator* neural_network_clone(const Estimator *self) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;

    int *hidden_sizes = malloc((nn->n_layer_sizes - 2) * sizeof(int));
    for (int i = 1; i < nn->n_layer_sizes - 1; i++) {
        hidden_sizes[i - 1] = nn->layer_sizes[i];
    }

    NeuralNetwork *clone = neural_network_create_full(
        hidden_sizes, nn->n_layer_sizes - 2,
        nn->hidden_activation, nn->output_activation,
        nn->loss_type, nn->optimizer,
        nn->learning_rate, nn->max_epochs, nn->batch_size,
        nn->momentum, nn->l2_reg
    );

    free(hidden_sizes);
    return (Estimator*)clone;
}

void neural_network_free(Estimator *self) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn) return;

    for (int i = 0; i < nn->n_layers; i++) {
        layer_free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn->layer_sizes);
    training_history_free(nn->base.history);
    free(nn);
}

void neural_network_print_summary(const Estimator *self) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;

    printf("\n=== Neural Network Summary ===\n");
    printf("Fitted: %s\n", nn->base.is_fitted ? "Yes" : "No");
    printf("Task: %s\n", nn->base.task == TASK_CLASSIFICATION ? "Classification" : "Regression");

    printf("\nArchitecture:\n");
    for (int i = 0; i < nn->n_layer_sizes; i++) {
        if (i == 0) printf("  Input:  %d neurons\n", nn->layer_sizes[i]);
        else if (i == nn->n_layer_sizes - 1) printf("  Output: %d neurons\n", nn->layer_sizes[i]);
        else printf("  Hidden: %d neurons\n", nn->layer_sizes[i]);
    }

    printf("\nHyperparameters:\n");
    printf("  Learning rate: %.6f\n", nn->learning_rate);
    printf("  Epochs: %d\n", nn->max_epochs);
    printf("  Batch size: %d\n", nn->batch_size);

    const char *optimizer_names[] = {"SGD", "Momentum", "Adam"};
    printf("  Optimizer: %s\n", optimizer_names[nn->optimizer]);

    if (nn->base.is_fitted) {
        int total_params = 0;
        for (int i = 0; i < nn->n_layers; i++) {
            total_params += nn->layers[i]->weights->rows * nn->layers[i]->weights->cols;
            total_params += nn->layers[i]->biases->cols;
        }
        printf("\nTotal parameters: %d\n", total_params);
    }
    printf("==============================\n\n");
}

/* ============================================
 * MLP Classifier/Regressor shortcuts
 * ============================================ */

MLPClassifier* mlp_classifier_create(const int *hidden_layer_sizes, int n_hidden) {
    NeuralNetwork *nn = neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        ACTIVATION_RELU, ACTIVATION_SOFTMAX,
        LOSS_CROSS_ENTROPY, OPTIMIZER_ADAM,
        0.001, 200, 32, 0.9, 0.0001
    );
    nn->base.task = TASK_CLASSIFICATION;
    return nn;
}

MLPRegressor* mlp_regressor_create(const int *hidden_layer_sizes, int n_hidden) {
    NeuralNetwork *nn = neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        ACTIVATION_RELU, ACTIVATION_IDENTITY,
        LOSS_MSE, OPTIMIZER_ADAM,
        0.001, 200, 32, 0.9, 0.0001
    );
    nn->base.task = TASK_REGRESSION;
    return nn;
}


/* --- validation.c --- */
/**
 * validation.c - Cross-validation implementation
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Helper functions
 * ============================================ */

Matrix* matrix_get_rows(const Matrix *m, const size_t *indices, size_t n_indices) {
    if (!m || !indices || n_indices == 0) return NULL;

    Matrix *result = matrix_alloc(n_indices, m->cols);
    if (!result) return NULL;

    for (size_t i = 0; i < n_indices; i++) {
        size_t src_row = indices[i];
        if (src_row >= m->rows) {
            matrix_free(result);
            return NULL;
        }
        for (size_t j = 0; j < m->cols; j++) {
            result->data[i * m->cols + j] = m->data[src_row * m->cols + j];
        }
    }

    return result;
}

static double array_mean(const double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum / n;
}

static double array_std(const double *arr, int n) {
    double m = array_mean(arr, n);
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = arr[i] - m;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

/* ============================================
 * K-Fold Cross-Validation
 * ============================================ */

KFold* kfold_create(int n_splits, int shuffle, unsigned int seed) {
    if (n_splits < 2) return NULL;

    KFold *kf = calloc(1, sizeof(KFold));
    if (!kf) return NULL;

    kf->n_splits = n_splits;
    kf->shuffle = shuffle;
    kf->seed = seed;
    kf->n_samples = 0;
    kf->indices = NULL;
    kf->current_fold = 0;

    return kf;
}

void kfold_init(KFold *kf, size_t n_samples) {
    if (!kf || n_samples == 0) return;

    kf->n_samples = n_samples;

    // Free old indices if any
    free(kf->indices);

    // Allocate indices
    kf->indices = malloc(n_samples * sizeof(size_t));
    if (!kf->indices) return;

    // Initialize indices
    for (size_t i = 0; i < n_samples; i++) {
        kf->indices[i] = i;
    }

    // Shuffle if requested
    if (kf->shuffle) {
        rand_seed(kf->seed);
        shuffle_indices(kf->indices, n_samples);
    }
}

FoldIndices kfold_get_fold(const KFold *kf, int fold) {
    FoldIndices fi = {NULL, 0, NULL, 0};

    if (!kf || !kf->indices || fold < 0 || fold >= kf->n_splits) {
        return fi;
    }

    size_t n = kf->n_samples;
    int k = kf->n_splits;

    // Calculate fold boundaries
    size_t fold_size = n / k;
    size_t remainder = n % k;

    size_t test_start = fold * fold_size + (fold < (int)remainder ? fold : remainder);
    size_t test_end = test_start + fold_size + (fold < (int)remainder ? 1 : 0);
    size_t n_test = test_end - test_start;
    size_t n_train = n - n_test;

    // Allocate arrays
    fi.test_indices = malloc(n_test * sizeof(size_t));
    fi.train_indices = malloc(n_train * sizeof(size_t));

    if (!fi.test_indices || !fi.train_indices) {
        free(fi.test_indices);
        free(fi.train_indices);
        fi.test_indices = NULL;
        fi.train_indices = NULL;
        return fi;
    }

    fi.n_test = n_test;
    fi.n_train = n_train;

    // Fill test indices
    for (size_t i = 0; i < n_test; i++) {
        fi.test_indices[i] = kf->indices[test_start + i];
    }

    // Fill train indices (everything except test)
    size_t train_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < test_start || i >= test_end) {
            fi.train_indices[train_idx++] = kf->indices[i];
        }
    }

    return fi;
}

void fold_indices_free(FoldIndices *fi) {
    if (fi) {
        free(fi->train_indices);
        free(fi->test_indices);
        fi->train_indices = NULL;
        fi->test_indices = NULL;
        fi->n_train = 0;
        fi->n_test = 0;
    }
}

void kfold_free(KFold *kf) {
    if (kf) {
        free(kf->indices);
        free(kf);
    }
}

/* ============================================
 * Stratified K-Fold
 * ============================================ */

StratifiedKFold* stratified_kfold_create(int n_splits, int shuffle, unsigned int seed) {
    if (n_splits < 2) return NULL;

    StratifiedKFold *skf = calloc(1, sizeof(StratifiedKFold));
    if (!skf) return NULL;

    skf->n_splits = n_splits;
    skf->shuffle = shuffle;
    skf->seed = seed;
    skf->n_samples = 0;
    skf->indices = NULL;
    skf->y = NULL;
    skf->n_classes = 0;

    return skf;
}

void stratified_kfold_init(StratifiedKFold *skf, const Matrix *y) {
    if (!skf || !y) return;

    skf->y = y;
    skf->n_samples = y->rows;

    // Find unique classes and count them
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    skf->n_classes = max_class + 1;

    // Count samples per class
    size_t *class_counts = calloc(skf->n_classes, sizeof(size_t));
    if (!class_counts) return;

    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        class_counts[label]++;
    }

    // Create indices array grouped by class
    size_t **class_indices = malloc(skf->n_classes * sizeof(size_t*));
    size_t *class_positions = calloc(skf->n_classes, sizeof(size_t));

    if (!class_indices || !class_positions) {
        free(class_counts);
        free(class_indices);
        free(class_positions);
        return;
    }

    for (int c = 0; c < skf->n_classes; c++) {
        class_indices[c] = malloc(class_counts[c] * sizeof(size_t));
        if (!class_indices[c]) {
            for (int j = 0; j < c; j++) free(class_indices[j]);
            free(class_indices);
            free(class_counts);
            free(class_positions);
            return;
        }
    }

    // Fill class indices
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        class_indices[label][class_positions[label]++] = i;
    }

    // Shuffle within each class if requested
    if (skf->shuffle) {
        rand_seed(skf->seed);
        for (int c = 0; c < skf->n_classes; c++) {
            shuffle_indices(class_indices[c], class_counts[c]);
        }
    }

    // Interleave indices from each class
    free(skf->indices);
    skf->indices = malloc(skf->n_samples * sizeof(size_t));
    if (!skf->indices) {
        for (int c = 0; c < skf->n_classes; c++) free(class_indices[c]);
        free(class_indices);
        free(class_counts);
        free(class_positions);
        return;
    }

    size_t idx = 0;
    memset(class_positions, 0, skf->n_classes * sizeof(size_t));

    // Round-robin from each class
    int done = 0;
    while (!done) {
        done = 1;
        for (int c = 0; c < skf->n_classes; c++) {
            if (class_positions[c] < class_counts[c]) {
                skf->indices[idx++] = class_indices[c][class_positions[c]++];
                done = 0;
            }
        }
    }

    // Cleanup
    for (int c = 0; c < skf->n_classes; c++) free(class_indices[c]);
    free(class_indices);
    free(class_counts);
    free(class_positions);
}

FoldIndices stratified_kfold_get_fold(const StratifiedKFold *skf, int fold) {
    // Similar to kfold but uses stratified indices
    FoldIndices fi = {NULL, 0, NULL, 0};

    if (!skf || !skf->indices || fold < 0 || fold >= skf->n_splits) {
        return fi;
    }

    size_t n = skf->n_samples;
    int k = skf->n_splits;

    size_t fold_size = n / k;
    size_t remainder = n % k;

    size_t test_start = fold * fold_size + (fold < (int)remainder ? fold : remainder);
    size_t test_end = test_start + fold_size + (fold < (int)remainder ? 1 : 0);
    size_t n_test = test_end - test_start;
    size_t n_train = n - n_test;

    fi.test_indices = malloc(n_test * sizeof(size_t));
    fi.train_indices = malloc(n_train * sizeof(size_t));

    if (!fi.test_indices || !fi.train_indices) {
        free(fi.test_indices);
        free(fi.train_indices);
        fi.test_indices = NULL;
        fi.train_indices = NULL;
        return fi;
    }

    fi.n_test = n_test;
    fi.n_train = n_train;

    for (size_t i = 0; i < n_test; i++) {
        fi.test_indices[i] = skf->indices[test_start + i];
    }

    size_t train_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < test_start || i >= test_end) {
            fi.train_indices[train_idx++] = skf->indices[i];
        }
    }

    return fi;
}

void stratified_kfold_free(StratifiedKFold *skf) {
    if (skf) {
        free(skf->indices);
        free(skf);
    }
}

/* ============================================
 * Cross-validation scoring
 * ============================================ */

CrossValResults* cross_val_score(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
) {
    if (!estimator || !X || !y || n_splits < 2) return NULL;

    CrossValResults *results = calloc(1, sizeof(CrossValResults));
    if (!results) return NULL;

    results->n_splits = n_splits;
    results->test_scores = malloc(n_splits * sizeof(double));
    results->train_scores = malloc(n_splits * sizeof(double));
    results->fit_times = malloc(n_splits * sizeof(double));
    results->score_times = malloc(n_splits * sizeof(double));

    if (!results->test_scores || !results->train_scores ||
        !results->fit_times || !results->score_times) {
        cross_val_results_free(results);
        return NULL;
    }

    // Create K-Fold
    KFold *kf = kfold_create(n_splits, shuffle, seed);
    if (!kf) {
        cross_val_results_free(results);
        return NULL;
    }
    kfold_init(kf, X->rows);

    // Iterate over folds
    for (int fold = 0; fold < n_splits; fold++) {
        FoldIndices fi = kfold_get_fold(kf, fold);
        if (!fi.train_indices || !fi.test_indices) {
            fold_indices_free(&fi);
            continue;
        }

        // Get train/test data for this fold
        Matrix *X_train = matrix_get_rows(X, fi.train_indices, fi.n_train);
        Matrix *y_train = matrix_get_rows(y, fi.train_indices, fi.n_train);
        Matrix *X_test = matrix_get_rows(X, fi.test_indices, fi.n_test);
        Matrix *y_test = matrix_get_rows(y, fi.test_indices, fi.n_test);

        if (!X_train || !y_train || !X_test || !y_test) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        // Clone estimator
        Estimator *clone = estimator->clone(estimator);
        if (!clone) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        // Fit
        clock_t fit_start = clock();
        clone->fit(clone, X_train, y_train);
        clock_t fit_end = clock();
        results->fit_times[fold] = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

        // Score
        clock_t score_start = clock();
        results->train_scores[fold] = clone->score(clone, X_train, y_train);
        results->test_scores[fold] = clone->score(clone, X_test, y_test);
        clock_t score_end = clock();
        results->score_times[fold] = (double)(score_end - score_start) / CLOCKS_PER_SEC;

        // Cleanup
        clone->free(clone);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        fold_indices_free(&fi);
    }

    kfold_free(kf);

    // Calculate summary statistics
    results->mean_test_score = array_mean(results->test_scores, n_splits);
    results->std_test_score = array_std(results->test_scores, n_splits);
    results->mean_train_score = array_mean(results->train_scores, n_splits);
    results->std_train_score = array_std(results->train_scores, n_splits);

    return results;
}

CrossValResults* cross_val_score_stratified(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    int n_splits,
    int shuffle,
    unsigned int seed
) {
    if (!estimator || !X || !y || n_splits < 2) return NULL;

    CrossValResults *results = calloc(1, sizeof(CrossValResults));
    if (!results) return NULL;

    results->n_splits = n_splits;
    results->test_scores = malloc(n_splits * sizeof(double));
    results->train_scores = malloc(n_splits * sizeof(double));
    results->fit_times = malloc(n_splits * sizeof(double));
    results->score_times = malloc(n_splits * sizeof(double));

    if (!results->test_scores || !results->train_scores ||
        !results->fit_times || !results->score_times) {
        cross_val_results_free(results);
        return NULL;
    }

    // Create Stratified K-Fold
    StratifiedKFold *skf = stratified_kfold_create(n_splits, shuffle, seed);
    if (!skf) {
        cross_val_results_free(results);
        return NULL;
    }
    stratified_kfold_init(skf, y);

    // Iterate over folds
    for (int fold = 0; fold < n_splits; fold++) {
        FoldIndices fi = stratified_kfold_get_fold(skf, fold);
        if (!fi.train_indices || !fi.test_indices) {
            fold_indices_free(&fi);
            continue;
        }

        Matrix *X_train = matrix_get_rows(X, fi.train_indices, fi.n_train);
        Matrix *y_train = matrix_get_rows(y, fi.train_indices, fi.n_train);
        Matrix *X_test = matrix_get_rows(X, fi.test_indices, fi.n_test);
        Matrix *y_test = matrix_get_rows(y, fi.test_indices, fi.n_test);

        if (!X_train || !y_train || !X_test || !y_test) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        Estimator *clone = estimator->clone(estimator);
        if (!clone) {
            matrix_free(X_train);
            matrix_free(y_train);
            matrix_free(X_test);
            matrix_free(y_test);
            fold_indices_free(&fi);
            continue;
        }

        clock_t fit_start = clock();
        clone->fit(clone, X_train, y_train);
        clock_t fit_end = clock();
        results->fit_times[fold] = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

        clock_t score_start = clock();
        results->train_scores[fold] = clone->score(clone, X_train, y_train);
        results->test_scores[fold] = clone->score(clone, X_test, y_test);
        clock_t score_end = clock();
        results->score_times[fold] = (double)(score_end - score_start) / CLOCKS_PER_SEC;

        clone->free(clone);
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
        fold_indices_free(&fi);
    }

    stratified_kfold_free(skf);

    results->mean_test_score = array_mean(results->test_scores, n_splits);
    results->std_test_score = array_std(results->test_scores, n_splits);
    results->mean_train_score = array_mean(results->train_scores, n_splits);
    results->std_train_score = array_std(results->train_scores, n_splits);

    return results;
}

void cross_val_results_print(const CrossValResults *results) {
    if (!results) return;

    printf("\n=== Cross-Validation Results ===\n");
    printf("Number of folds: %d\n\n", results->n_splits);

    printf("%-8s %-15s %-15s %-12s\n", "Fold", "Train Score", "Test Score", "Fit Time");
    printf("----------------------------------------------------\n");

    for (int i = 0; i < results->n_splits; i++) {
        printf("%-8d %-15.6f %-15.6f %-12.4f\n",
               i + 1,
               results->train_scores[i],
               results->test_scores[i],
               results->fit_times[i]);
    }

    printf("----------------------------------------------------\n");
    printf("Mean test score:  %.6f (+/- %.6f)\n",
           results->mean_test_score, results->std_test_score * 2);
    printf("Mean train score: %.6f (+/- %.6f)\n",
           results->mean_train_score, results->std_train_score * 2);
    printf("================================\n\n");
}

void cross_val_results_free(CrossValResults *results) {
    if (results) {
        free(results->test_scores);
        free(results->train_scores);
        free(results->fit_times);
        free(results->score_times);
        free(results);
    }
}

/* ============================================
 * Leave-One-Out Cross-Validation
 * ============================================ */

CrossValResults* leave_one_out_cv(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y
) {
    return cross_val_score(estimator, X, y, (int)X->rows, 0, 0);
}


/* --- feature_selection.c --- */
/**
 * feature_selection.c - Feature Selection Utilities
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Statistical Helper Functions
 * ============================================ */

static double compute_mean(const double *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

static double compute_variance(const double *data, size_t n, double mean) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / n;
}

// Quickselect to find k-th smallest element index
static size_t partition_indices(double *scores, size_t *indices, size_t left, size_t right, size_t pivot_idx) {
    double pivot_val = scores[pivot_idx];

    // Swap pivot to end
    double tmp_s = scores[pivot_idx]; scores[pivot_idx] = scores[right]; scores[right] = tmp_s;
    size_t tmp_i = indices[pivot_idx]; indices[pivot_idx] = indices[right]; indices[right] = tmp_i;

    size_t store_idx = left;
    for (size_t i = left; i < right; i++) {
        if (scores[i] > pivot_val) {  // Descending order (higher scores first)
            tmp_s = scores[i]; scores[i] = scores[store_idx]; scores[store_idx] = tmp_s;
            tmp_i = indices[i]; indices[i] = indices[store_idx]; indices[store_idx] = tmp_i;
            store_idx++;
        }
    }

    // Swap pivot back
    tmp_s = scores[store_idx]; scores[store_idx] = scores[right]; scores[right] = tmp_s;
    tmp_i = indices[store_idx]; indices[store_idx] = indices[right]; indices[right] = tmp_i;

    return store_idx;
}

static void select_top_k_indices(double *scores, size_t *indices, size_t n, size_t k) {
    // Partial quicksort to get top k (descending)
    size_t left = 0, right = n - 1;

    while (left < right) {
        size_t pivot_idx = left + (right - left) / 2;
        pivot_idx = partition_indices(scores, indices, left, right, pivot_idx);

        if (pivot_idx == k - 1) {
            break;
        } else if (pivot_idx > k - 1) {
            right = pivot_idx - 1;
        } else {
            left = pivot_idx + 1;
        }
    }
}

/* ============================================
 * Scoring Functions
 * ============================================ */

int f_classif(const Matrix *X, const Matrix *y, double *f_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Count samples per class
    size_t *class_counts = calloc(n_classes, sizeof(size_t));
    for (size_t i = 0; i < n_samples; i++) {
        class_counts[(int)y->data[i]]++;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Compute overall mean
        double grand_mean = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            grand_mean += X->data[i * n_features + j];
        }
        grand_mean /= n_samples;

        // Compute class means
        double *class_means = calloc(n_classes, sizeof(double));
        for (size_t i = 0; i < n_samples; i++) {
            int label = (int)y->data[i];
            class_means[label] += X->data[i * n_features + j];
        }
        for (int c = 0; c < n_classes; c++) {
            if (class_counts[c] > 0) {
                class_means[c] /= class_counts[c];
            }
        }

        // Between-class variance (SSB)
        double ss_between = 0.0;
        for (int c = 0; c < n_classes; c++) {
            double diff = class_means[c] - grand_mean;
            ss_between += class_counts[c] * diff * diff;
        }

        // Within-class variance (SSW)
        double ss_within = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int label = (int)y->data[i];
            double diff = X->data[i * n_features + j] - class_means[label];
            ss_within += diff * diff;
        }

        free(class_means);

        // F-statistic = (SSB / df_between) / (SSW / df_within)
        int df_between = n_classes - 1;
        int df_within = n_samples - n_classes;

        if (df_within > 0 && ss_within > 0) {
            double ms_between = ss_between / df_between;
            double ms_within = ss_within / df_within;
            f_values[j] = ms_between / ms_within;
        } else {
            f_values[j] = 0.0;
        }

        // p-value approximation (simplified - would need F-distribution CDF)
        if (p_values) {
            p_values[j] = 0.0;  // Placeholder - full implementation needs F-dist CDF
        }
    }

    free(class_counts);
    return 0;
}

int f_regression(const Matrix *X, const Matrix *y, double *f_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Mean of y
    double y_mean = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        y_mean += y->data[i];
    }
    y_mean /= n_samples;

    // Variance of y
    double y_var = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        double diff = y->data[i] - y_mean;
        y_var += diff * diff;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Mean of feature
        double x_mean = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            x_mean += X->data[i * n_features + j];
        }
        x_mean /= n_samples;

        // Correlation coefficient
        double cov = 0.0;
        double x_var = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            double x_diff = X->data[i * n_features + j] - x_mean;
            double y_diff = y->data[i] - y_mean;
            cov += x_diff * y_diff;
            x_var += x_diff * x_diff;
        }

        double r = 0.0;
        if (x_var > 0 && y_var > 0) {
            r = cov / sqrt(x_var * y_var);
        }

        // F-statistic from correlation: F = r² * (n-2) / (1 - r²)
        double r2 = r * r;
        if (r2 < 1.0 && n_samples > 2) {
            f_values[j] = r2 * (n_samples - 2) / (1.0 - r2);
        } else {
            f_values[j] = 0.0;
        }

        if (p_values) {
            p_values[j] = 0.0;  // Placeholder
        }
    }

    return 0;
}

int chi2(const Matrix *X, const Matrix *y, double *chi2_values, double *p_values) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Sum per class
    double *class_sums = calloc(n_classes, sizeof(double));
    double total_sum = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++) {
            double val = X->data[i * n_features + j];
            if (val < 0) {
                free(class_sums);
                return -1;  // Chi2 requires non-negative features
            }
            class_sums[(int)y->data[i]] += val;
            total_sum += val;
        }
    }

    for (size_t j = 0; j < n_features; j++) {
        // Feature sum
        double feature_sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            feature_sum += X->data[i * n_features + j];
        }

        // Chi-squared for this feature
        double chi2_val = 0.0;
        for (int c = 0; c < n_classes; c++) {
            // Observed
            double observed = 0.0;
            for (size_t i = 0; i < n_samples; i++) {
                if ((int)y->data[i] == c) {
                    observed += X->data[i * n_features + j];
                }
            }

            // Expected = (feature_sum * class_sum) / total
            double expected = 0.0;
            if (total_sum > 0) {
                expected = (feature_sum * class_sums[c]) / total_sum;
            }

            if (expected > 0) {
                chi2_val += (observed - expected) * (observed - expected) / expected;
            }
        }

        chi2_values[j] = chi2_val;
        if (p_values) p_values[j] = 0.0;
    }

    free(class_sums);
    return 0;
}

int mutual_info_classif(const Matrix *X, const Matrix *y, double *mi_values, int n_bins) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    if (n_bins <= 0) n_bins = 10;

    // Find number of classes
    int n_classes = 0;
    for (size_t i = 0; i < n_samples; i++) {
        int label = (int)y->data[i];
        if (label + 1 > n_classes) n_classes = label + 1;
    }

    // Class probabilities
    double *p_y = calloc(n_classes, sizeof(double));
    for (size_t i = 0; i < n_samples; i++) {
        p_y[(int)y->data[i]] += 1.0 / n_samples;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Find feature range
        double x_min = X->data[j];
        double x_max = X->data[j];
        for (size_t i = 1; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            if (val < x_min) x_min = val;
            if (val > x_max) x_max = val;
        }

        double bin_width = (x_max - x_min) / n_bins;
        if (bin_width < 1e-10) bin_width = 1.0;

        // Bin counts: p(x), p(x,y)
        double *p_x = calloc(n_bins, sizeof(double));
        double *p_xy = calloc(n_bins * n_classes, sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            int bin = (int)((val - x_min) / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;
            if (bin < 0) bin = 0;

            int label = (int)y->data[i];
            p_x[bin] += 1.0 / n_samples;
            p_xy[bin * n_classes + label] += 1.0 / n_samples;
        }

        // MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        double mi = 0.0;
        for (int b = 0; b < n_bins; b++) {
            for (int c = 0; c < n_classes; c++) {
                double pxy = p_xy[b * n_classes + c];
                double px = p_x[b];
                double py = p_y[c];

                if (pxy > 0 && px > 0 && py > 0) {
                    mi += pxy * log(pxy / (px * py));
                }
            }
        }

        mi_values[j] = mi > 0 ? mi : 0.0;

        free(p_x);
        free(p_xy);
    }

    free(p_y);
    return 0;
}

int mutual_info_regression(const Matrix *X, const Matrix *y, double *mi_values, int n_bins) {
    size_t n_samples = X->rows;
    size_t n_features = X->cols;

    if (n_bins <= 0) n_bins = 10;

    // Find y range
    double y_min = y->data[0];
    double y_max = y->data[0];
    for (size_t i = 1; i < n_samples; i++) {
        if (y->data[i] < y_min) y_min = y->data[i];
        if (y->data[i] > y_max) y_max = y->data[i];
    }
    double y_bin_width = (y_max - y_min) / n_bins;
    if (y_bin_width < 1e-10) y_bin_width = 1.0;

    // Bin y values
    int *y_bins = malloc(n_samples * sizeof(int));
    double *p_y = calloc(n_bins, sizeof(double));
    for (size_t i = 0; i < n_samples; i++) {
        int bin = (int)((y->data[i] - y_min) / y_bin_width);
        if (bin >= n_bins) bin = n_bins - 1;
        if (bin < 0) bin = 0;
        y_bins[i] = bin;
        p_y[bin] += 1.0 / n_samples;
    }

    for (size_t j = 0; j < n_features; j++) {
        // Find feature range
        double x_min = X->data[j];
        double x_max = X->data[j];
        for (size_t i = 1; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            if (val < x_min) x_min = val;
            if (val > x_max) x_max = val;
        }

        double bin_width = (x_max - x_min) / n_bins;
        if (bin_width < 1e-10) bin_width = 1.0;

        double *p_x = calloc(n_bins, sizeof(double));
        double *p_xy = calloc(n_bins * n_bins, sizeof(double));

        for (size_t i = 0; i < n_samples; i++) {
            double val = X->data[i * n_features + j];
            int x_bin = (int)((val - x_min) / bin_width);
            if (x_bin >= n_bins) x_bin = n_bins - 1;
            if (x_bin < 0) x_bin = 0;

            p_x[x_bin] += 1.0 / n_samples;
            p_xy[x_bin * n_bins + y_bins[i]] += 1.0 / n_samples;
        }

        // MI calculation
        double mi = 0.0;
        for (int bx = 0; bx < n_bins; bx++) {
            for (int by = 0; by < n_bins; by++) {
                double pxy = p_xy[bx * n_bins + by];
                double px = p_x[bx];
                double py = p_y[by];

                if (pxy > 0 && px > 0 && py > 0) {
                    mi += pxy * log(pxy / (px * py));
                }
            }
        }

        mi_values[j] = mi > 0 ? mi : 0.0;

        free(p_x);
        free(p_xy);
    }

    free(y_bins);
    free(p_y);
    return 0;
}

/* ============================================
 * SelectKBest Implementation
 * ============================================ */

SelectKBest* select_k_best_create(ScoreFunction score_func, int k) {
    SelectKBest *skb = calloc(1, sizeof(SelectKBest));
    if (!skb) return NULL;

    skb->base.type = MODEL_FEATURE_SELECTOR;
    skb->base.task = TASK_TRANSFORMATION;
    skb->base.is_fitted = 0;
    skb->base.verbose = VERBOSE_SILENT;

    skb->base.fit = select_k_best_fit;
    skb->base.predict = NULL;
    skb->base.predict_proba = NULL;
    skb->base.transform = select_k_best_transform;
    skb->base.score = NULL;
    skb->base.clone = select_k_best_clone;
    skb->base.free = select_k_best_free;
    skb->base.save = NULL;
    skb->base.load = NULL;
    skb->base.print_summary = select_k_best_print_summary;

    skb->score_func = score_func;
    skb->k = k;
    skb->scores_ = NULL;
    skb->pvalues_ = NULL;
    skb->support_ = NULL;

    return skb;
}

Estimator* select_k_best_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    SelectKBest *skb = (SelectKBest*)self;

    skb->n_features_ = X->cols;
    int k = skb->k;
    if (k <= 0 || k > (int)X->cols) {
        k = X->cols;
    }
    skb->n_features_selected_ = k;

    // Allocate
    skb->scores_ = malloc(X->cols * sizeof(double));
    skb->pvalues_ = malloc(X->cols * sizeof(double));
    skb->support_ = calloc(X->cols, sizeof(int));

    // Compute scores
    switch (skb->score_func) {
        case SCORE_F_CLASSIF:
            f_classif(X, y, skb->scores_, skb->pvalues_);
            break;
        case SCORE_F_REGRESSION:
            f_regression(X, y, skb->scores_, skb->pvalues_);
            break;
        case SCORE_MUTUAL_INFO_CLASSIF:
            mutual_info_classif(X, y, skb->scores_, 10);
            break;
        case SCORE_MUTUAL_INFO_REGRESSION:
            mutual_info_regression(X, y, skb->scores_, 10);
            break;
        case SCORE_CHI2:
            chi2(X, y, skb->scores_, skb->pvalues_);
            break;
    }

    // Select top k features
    double *scores_copy = malloc(X->cols * sizeof(double));
    size_t *indices = malloc(X->cols * sizeof(size_t));
    for (size_t j = 0; j < X->cols; j++) {
        scores_copy[j] = skb->scores_[j];
        indices[j] = j;
    }

    select_top_k_indices(scores_copy, indices, X->cols, k);

    // Mark selected features
    for (int i = 0; i < k; i++) {
        skb->support_[indices[i]] = 1;
    }

    free(scores_copy);
    free(indices);

    skb->base.is_fitted = 1;

    if (skb->base.verbose >= VERBOSE_MINIMAL) {
        printf("SelectKBest: selected %d/%d features\n", k, skb->n_features_);
    }

    return self;
}

Matrix* select_k_best_transform(const Estimator *self, const Matrix *X) {
    const SelectKBest *skb = (const SelectKBest*)self;

    if (!skb->base.is_fitted) return NULL;

    Matrix *X_new = matrix_alloc(X->rows, skb->n_features_selected_);
    if (!X_new) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        size_t new_j = 0;
        for (int j = 0; j < skb->n_features_; j++) {
            if (skb->support_[j]) {
                X_new->data[i * skb->n_features_selected_ + new_j] =
                    X->data[i * X->cols + j];
                new_j++;
            }
        }
    }

    return X_new;
}

const double* select_k_best_scores(const SelectKBest *skb) {
    return skb ? skb->scores_ : NULL;
}

const int* select_k_best_get_support(const SelectKBest *skb) {
    return skb ? skb->support_ : NULL;
}

Estimator* select_k_best_clone(const Estimator *self) {
    const SelectKBest *skb = (const SelectKBest*)self;
    return (Estimator*)select_k_best_create(skb->score_func, skb->k);
}

void select_k_best_free(Estimator *self) {
    SelectKBest *skb = (SelectKBest*)self;
    if (!skb) return;

    free(skb->scores_);
    free(skb->pvalues_);
    free(skb->support_);
    free(skb);
}

void select_k_best_print_summary(const Estimator *self) {
    const SelectKBest *skb = (const SelectKBest*)self;

    const char *score_names[] = {
        "f_classif", "f_regression", "mutual_info_classif",
        "mutual_info_regression", "chi2"
    };

    printf("\n=== SelectKBest Summary ===\n");
    printf("Fitted: %s\n", skb->base.is_fitted ? "Yes" : "No");
    printf("Score function: %s\n", score_names[skb->score_func]);
    printf("k: %d\n", skb->k);

    if (skb->base.is_fitted) {
        printf("Features selected: %d/%d\n", skb->n_features_selected_, skb->n_features_);
        printf("\nFeature scores:\n");
        for (int j = 0; j < skb->n_features_ && j < 10; j++) {
            printf("  [%d] %.4f %s\n", j, skb->scores_[j],
                   skb->support_[j] ? "(selected)" : "");
        }
        if (skb->n_features_ > 10) {
            printf("  ... (%d more)\n", skb->n_features_ - 10);
        }
    }
    printf("===========================\n\n");
}

/* ============================================
 * VarianceThreshold Implementation
 * ============================================ */

VarianceThreshold* variance_threshold_create(double threshold) {
    VarianceThreshold *vt = calloc(1, sizeof(VarianceThreshold));
    if (!vt) return NULL;

    vt->base.type = MODEL_FEATURE_SELECTOR;
    vt->base.task = TASK_TRANSFORMATION;
    vt->base.is_fitted = 0;
    vt->base.verbose = VERBOSE_SILENT;

    vt->base.fit = variance_threshold_fit;
    vt->base.predict = NULL;
    vt->base.predict_proba = NULL;
    vt->base.transform = variance_threshold_transform;
    vt->base.score = NULL;
    vt->base.clone = variance_threshold_clone;
    vt->base.free = variance_threshold_free;
    vt->base.save = NULL;
    vt->base.load = NULL;
    vt->base.print_summary = NULL;

    vt->threshold = threshold;
    vt->variances_ = NULL;
    vt->support_ = NULL;

    return vt;
}

Estimator* variance_threshold_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;
    VarianceThreshold *vt = (VarianceThreshold*)self;

    vt->n_features_ = X->cols;
    vt->variances_ = malloc(X->cols * sizeof(double));
    vt->support_ = calloc(X->cols, sizeof(int));
    vt->n_features_selected_ = 0;

    for (size_t j = 0; j < X->cols; j++) {
        // Compute mean
        double mean = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            mean += X->data[i * X->cols + j];
        }
        mean /= X->rows;

        // Compute variance
        double var = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            double diff = X->data[i * X->cols + j] - mean;
            var += diff * diff;
        }
        var /= X->rows;

        vt->variances_[j] = var;
        if (var > vt->threshold) {
            vt->support_[j] = 1;
            vt->n_features_selected_++;
        }
    }

    vt->base.is_fitted = 1;

    if (vt->base.verbose >= VERBOSE_MINIMAL) {
        printf("VarianceThreshold: kept %d/%d features (threshold=%.4f)\n",
               vt->n_features_selected_, vt->n_features_, vt->threshold);
    }

    return self;
}

Matrix* variance_threshold_transform(const Estimator *self, const Matrix *X) {
    const VarianceThreshold *vt = (const VarianceThreshold*)self;

    if (!vt->base.is_fitted) return NULL;
    if (vt->n_features_selected_ == 0) return NULL;

    Matrix *X_new = matrix_alloc(X->rows, vt->n_features_selected_);
    if (!X_new) return NULL;

    for (size_t i = 0; i < X->rows; i++) {
        size_t new_j = 0;
        for (int j = 0; j < vt->n_features_; j++) {
            if (vt->support_[j]) {
                X_new->data[i * vt->n_features_selected_ + new_j] =
                    X->data[i * X->cols + j];
                new_j++;
            }
        }
    }

    return X_new;
}

const double* variance_threshold_variances(const VarianceThreshold *vt) {
    return vt ? vt->variances_ : NULL;
}

const int* variance_threshold_get_support(const VarianceThreshold *vt) {
    return vt ? vt->support_ : NULL;
}

Estimator* variance_threshold_clone(const Estimator *self) {
    const VarianceThreshold *vt = (const VarianceThreshold*)self;
    return (Estimator*)variance_threshold_create(vt->threshold);
}

void variance_threshold_free(Estimator *self) {
    VarianceThreshold *vt = (VarianceThreshold*)self;
    if (!vt) return;

    free(vt->variances_);
    free(vt->support_);
    free(vt);
}

/* ============================================
 * Utility Functions
 * ============================================ */

int get_feature_importances(const Estimator *estimator, double *importances) {
    if (!estimator || !estimator->is_fitted) return -1;

    switch (estimator->type) {
        case MODEL_LINEAR_REGRESSION: {
            // Use absolute coefficient values
            // Need to access internal weights - simplified version
            return -1;  // Would need access to internal structure
        }
        case MODEL_DECISION_TREE:
        case MODEL_RANDOM_FOREST:
            // Would need to compute from tree structure
            return -1;
        default:
            return -1;
    }
}


/* --- ensemble.c --- */
/**
 * ensemble.c - Random Forest implementation
 */





#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Bootstrap sampling
 * ============================================ */

static void bootstrap_sample(size_t n_samples, size_t *indices, size_t *oob_mask, unsigned int seed) {
    rand_seed(seed);

    // Initialize OOB mask to 1 (out of bag)
    for (size_t i = 0; i < n_samples; i++) {
        oob_mask[i] = 1;
    }

    // Sample with replacement
    for (size_t i = 0; i < n_samples; i++) {
        indices[i] = (size_t)(rand_uniform() * n_samples);
        oob_mask[indices[i]] = 0;  // Mark as in bag
    }
}

/* ============================================
 * Random Forest Classifier
 * ============================================ */

RandomForestClassifier* random_forest_classifier_create(int n_estimators) {
    return random_forest_classifier_create_full(n_estimators, 10, 2, 1, 0, 1, 42);
}

RandomForestClassifier* random_forest_classifier_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
) {
    RandomForestClassifier *rf = calloc(1, sizeof(RandomForestClassifier));
    if (!rf) return NULL;

    rf->base.type = MODEL_RANDOM_FOREST;
    rf->base.task = TASK_CLASSIFICATION;
    rf->base.is_fitted = 0;
    rf->base.verbose = VERBOSE_SILENT;

    rf->base.fit = random_forest_classifier_fit;
    rf->base.predict = random_forest_classifier_predict;
    rf->base.predict_proba = random_forest_classifier_predict_proba;
    rf->base.transform = NULL;
    rf->base.score = random_forest_classifier_score;
    rf->base.clone = random_forest_classifier_clone;
    rf->base.free = random_forest_classifier_free;
    rf->base.save = NULL;
    rf->base.load = NULL;
    rf->base.print_summary = random_forest_classifier_print_summary;

    rf->n_estimators = n_estimators;
    rf->max_depth = max_depth;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf = min_samples_leaf;
    rf->max_features = max_features;
    rf->bootstrap = bootstrap;
    rf->seed = seed;

    rf->trees = NULL;
    rf->n_classes = 0;
    rf->oob_score_ = 0.0;

    return rf;
}

Estimator* random_forest_classifier_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    RandomForestClassifier *rf = (RandomForestClassifier*)self;
    clock_t start = clock();

    size_t n_samples = X->rows;
    rf->n_features = X->cols;

    // Determine n_classes
    int max_class = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int label = (int)y->data[i];
        if (label > max_class) max_class = label;
    }
    rf->n_classes = max_class + 1;

    // Determine max_features
    int max_feat = rf->max_features;
    if (max_feat <= 0) {
        max_feat = (int)sqrt((double)rf->n_features);
        if (max_feat < 1) max_feat = 1;
    }

    // Allocate trees
    rf->trees = malloc(rf->n_estimators * sizeof(DecisionTreeClassifier*));
    if (!rf->trees) return NULL;

    // OOB predictions for score calculation
    int *oob_counts = calloc(n_samples, sizeof(int));
    int *oob_predictions = calloc(n_samples * rf->n_classes, sizeof(int));

    if (rf->base.verbose >= VERBOSE_MINIMAL) {
        printf("Fitting %d trees...\n", rf->n_estimators);
    }

    for (int t = 0; t < rf->n_estimators; t++) {
        // Bootstrap sample
        size_t *indices = malloc(n_samples * sizeof(size_t));
        size_t *oob_mask = malloc(n_samples * sizeof(size_t));

        if (rf->bootstrap) {
            bootstrap_sample(n_samples, indices, oob_mask, rf->seed + t);
        } else {
            for (size_t i = 0; i < n_samples; i++) {
                indices[i] = i;
                oob_mask[i] = 0;
            }
        }

        // Create bootstrap sample
        Matrix *X_boot = matrix_get_rows(X, indices, n_samples);
        Matrix *y_boot = matrix_get_rows(y, indices, n_samples);

        // Create and fit tree
        rf->trees[t] = decision_tree_classifier_create_full(
            CRITERION_GINI,
            rf->max_depth,
            rf->min_samples_split,
            rf->min_samples_leaf,
            0.0
        );
        rf->trees[t]->max_features = max_feat;

        rf->trees[t]->base.fit((Estimator*)rf->trees[t], X_boot, y_boot);

        // OOB predictions
        if (rf->bootstrap) {
            for (size_t i = 0; i < n_samples; i++) {
                if (oob_mask[i]) {
                    Matrix *xi = matrix_alloc(1, X->cols);
                    for (size_t j = 0; j < X->cols; j++) {
                        xi->data[j] = X->data[i * X->cols + j];
                    }

                    Matrix *pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], xi);
                    if (pred) {
                        int pred_class = (int)pred->data[0];
                        oob_predictions[i * rf->n_classes + pred_class]++;
                        oob_counts[i]++;
                        matrix_free(pred);
                    }
                    matrix_free(xi);
                }
            }
        }

        matrix_free(X_boot);
        matrix_free(y_boot);
        free(indices);
        free(oob_mask);

        if (rf->base.verbose >= VERBOSE_PROGRESS && (t + 1) % 10 == 0) {
            printf("  [%d/%d] trees fitted\n", t + 1, rf->n_estimators);
        }
    }

    // Compute OOB score
    if (rf->bootstrap) {
        int correct = 0;
        int total = 0;
        for (size_t i = 0; i < n_samples; i++) {
            if (oob_counts[i] > 0) {
                int best_class = 0;
                int best_count = oob_predictions[i * rf->n_classes];
                for (int c = 1; c < rf->n_classes; c++) {
                    if (oob_predictions[i * rf->n_classes + c] > best_count) {
                        best_count = oob_predictions[i * rf->n_classes + c];
                        best_class = c;
                    }
                }
                if (best_class == (int)y->data[i]) {
                    correct++;
                }
                total++;
            }
        }
        rf->oob_score_ = total > 0 ? (double)correct / total : 0.0;
    }

    free(oob_counts);
    free(oob_predictions);

    rf->base.is_fitted = 1;

    if (rf->base.verbose >= VERBOSE_MINIMAL) {
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("RandomForest fitted in %.2f seconds\n", elapsed);
        if (rf->bootstrap) {
            printf("OOB score: %.4f\n", rf->oob_score_);
        }
    }

    return self;
}

Matrix* random_forest_classifier_predict(const Estimator *self, const Matrix *X) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;

    // Voting matrix
    int *votes = calloc(X->rows * rf->n_classes, sizeof(int));

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], X);
        if (tree_pred) {
            for (size_t i = 0; i < X->rows; i++) {
                int pred_class = (int)tree_pred->data[i];
                if (pred_class >= 0 && pred_class < rf->n_classes) {
                    votes[i * rf->n_classes + pred_class]++;
                }
            }
            matrix_free(tree_pred);
        }
    }

    // Majority vote
    for (size_t i = 0; i < X->rows; i++) {
        int best_class = 0;
        int best_count = votes[i * rf->n_classes];
        for (int c = 1; c < rf->n_classes; c++) {
            if (votes[i * rf->n_classes + c] > best_count) {
                best_count = votes[i * rf->n_classes + c];
                best_class = c;
            }
        }
        predictions->data[i] = best_class;
    }

    free(votes);
    return predictions;
}

Matrix* random_forest_classifier_predict_proba(const Estimator *self, const Matrix *X) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *proba = matrix_alloc(X->rows, rf->n_classes);
    if (!proba) return NULL;
    matrix_fill(proba, 0);

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_proba = rf->trees[t]->base.predict_proba((Estimator*)rf->trees[t], X);
        if (tree_proba) {
            for (size_t i = 0; i < proba->rows * proba->cols; i++) {
                proba->data[i] += tree_proba->data[i];
            }
            matrix_free(tree_proba);
        }
    }

    // Average
    for (size_t i = 0; i < proba->rows * proba->cols; i++) {
        proba->data[i] /= rf->n_estimators;
    }

    return proba;
}

double random_forest_classifier_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return classification_score_accuracy(self, X, y);
}

Estimator* random_forest_classifier_clone(const Estimator *self) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;
    return (Estimator*)random_forest_classifier_create_full(
        rf->n_estimators, rf->max_depth, rf->min_samples_split,
        rf->min_samples_leaf, rf->max_features, rf->bootstrap, rf->seed
    );
}

void random_forest_classifier_free(Estimator *self) {
    RandomForestClassifier *rf = (RandomForestClassifier*)self;
    if (!rf) return;

    if (rf->trees) {
        for (int t = 0; t < rf->n_estimators; t++) {
            if (rf->trees[t]) {
                rf->trees[t]->base.free((Estimator*)rf->trees[t]);
            }
        }
        free(rf->trees);
    }
    free(rf);
}

void random_forest_classifier_print_summary(const Estimator *self) {
    const RandomForestClassifier *rf = (const RandomForestClassifier*)self;

    printf("\n=== RandomForestClassifier Summary ===\n");
    printf("Fitted: %s\n", rf->base.is_fitted ? "Yes" : "No");
    printf("Estimators: %d\n", rf->n_estimators);
    printf("Max depth: %d\n", rf->max_depth);
    printf("Max features: %d\n", rf->max_features);
    printf("Bootstrap: %s\n", rf->bootstrap ? "Yes" : "No");

    if (rf->base.is_fitted) {
        printf("Classes: %d\n", rf->n_classes);
        printf("Features: %d\n", rf->n_features);
        if (rf->bootstrap) {
            printf("OOB Score: %.4f\n", rf->oob_score_);
        }
    }
    printf("======================================\n\n");
}

/* ============================================
 * Random Forest Regressor
 * ============================================ */

RandomForestRegressor* random_forest_regressor_create(int n_estimators) {
    return random_forest_regressor_create_full(n_estimators, 10, 2, 1, 0, 1, 42);
}

RandomForestRegressor* random_forest_regressor_create_full(
    int n_estimators,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    int max_features,
    int bootstrap,
    unsigned int seed
) {
    RandomForestRegressor *rf = calloc(1, sizeof(RandomForestRegressor));
    if (!rf) return NULL;

    rf->base.type = MODEL_RANDOM_FOREST;
    rf->base.task = TASK_REGRESSION;
    rf->base.is_fitted = 0;
    rf->base.verbose = VERBOSE_SILENT;

    rf->base.fit = random_forest_regressor_fit;
    rf->base.predict = random_forest_regressor_predict;
    rf->base.predict_proba = NULL;
    rf->base.transform = NULL;
    rf->base.score = random_forest_regressor_score;
    rf->base.clone = random_forest_regressor_clone;
    rf->base.free = random_forest_regressor_free;
    rf->base.save = NULL;
    rf->base.load = NULL;
    rf->base.print_summary = NULL;

    rf->n_estimators = n_estimators;
    rf->max_depth = max_depth;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf = min_samples_leaf;
    rf->max_features = max_features;
    rf->bootstrap = bootstrap;
    rf->seed = seed;

    rf->trees = NULL;
    rf->oob_score_ = 0.0;

    return rf;
}

Estimator* random_forest_regressor_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    RandomForestRegressor *rf = (RandomForestRegressor*)self;

    size_t n_samples = X->rows;
    rf->n_features = X->cols;

    int max_feat = rf->max_features;
    if (max_feat <= 0) {
        max_feat = rf->n_features;  // Use all features for regression
    }

    rf->trees = malloc(rf->n_estimators * sizeof(DecisionTreeRegressor*));
    if (!rf->trees) return NULL;

    for (int t = 0; t < rf->n_estimators; t++) {
        size_t *indices = malloc(n_samples * sizeof(size_t));
        size_t *oob_mask = malloc(n_samples * sizeof(size_t));

        if (rf->bootstrap) {
            bootstrap_sample(n_samples, indices, oob_mask, rf->seed + t);
        } else {
            for (size_t i = 0; i < n_samples; i++) {
                indices[i] = i;
            }
        }

        Matrix *X_boot = matrix_get_rows(X, indices, n_samples);
        Matrix *y_boot = matrix_get_rows(y, indices, n_samples);

        rf->trees[t] = decision_tree_regressor_create_full(
            CRITERION_MSE,
            rf->max_depth,
            rf->min_samples_split,
            rf->min_samples_leaf,
            0.0
        );
        rf->trees[t]->max_features = max_feat;

        rf->trees[t]->base.fit((Estimator*)rf->trees[t], X_boot, y_boot);

        matrix_free(X_boot);
        matrix_free(y_boot);
        free(indices);
        free(oob_mask);
    }

    rf->base.is_fitted = 1;
    return self;
}

Matrix* random_forest_regressor_predict(const Estimator *self, const Matrix *X) {
    const RandomForestRegressor *rf = (const RandomForestRegressor*)self;

    if (!rf->base.is_fitted) return NULL;

    Matrix *predictions = matrix_alloc(X->rows, 1);
    if (!predictions) return NULL;
    matrix_fill(predictions, 0);

    for (int t = 0; t < rf->n_estimators; t++) {
        Matrix *tree_pred = rf->trees[t]->base.predict((Estimator*)rf->trees[t], X);
        if (tree_pred) {
            for (size_t i = 0; i < X->rows; i++) {
                predictions->data[i] += tree_pred->data[i];
            }
            matrix_free(tree_pred);
        }
    }

    // Average
    for (size_t i = 0; i < X->rows; i++) {
        predictions->data[i] /= rf->n_estimators;
    }

    return predictions;
}

double random_forest_regressor_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    return regression_score_r2(self, X, y);
}

Estimator* random_forest_regressor_clone(const Estimator *self) {
    const RandomForestRegressor *rf = (const RandomForestRegressor*)self;
    return (Estimator*)random_forest_regressor_create_full(
        rf->n_estimators, rf->max_depth, rf->min_samples_split,
        rf->min_samples_leaf, rf->max_features, rf->bootstrap, rf->seed
    );
}

void random_forest_regressor_free(Estimator *self) {
    RandomForestRegressor *rf = (RandomForestRegressor*)self;
    if (!rf) return;

    if (rf->trees) {
        for (int t = 0; t < rf->n_estimators; t++) {
            if (rf->trees[t]) {
                rf->trees[t]->base.free((Estimator*)rf->trees[t]);
            }
        }
        free(rf->trees);
    }
    free(rf);
}


/* --- model_selection.c --- */
/**
 * model_selection.c - GridSearchCV and learning curves implementation
 */






#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Parameter Grid Implementation
 * ============================================ */

ParameterGrid* param_grid_create(int n_params) {
    ParameterGrid *grid = calloc(1, sizeof(ParameterGrid));
    if (!grid) return NULL;

    grid->params = calloc(n_params, sizeof(ParamSpec));
    if (!grid->params) {
        free(grid);
        return NULL;
    }
    grid->n_params = n_params;

    return grid;
}

int param_grid_add_int(ParameterGrid *grid, int index, const char *name,
                       const int *values, int n_values) {
    if (!grid || index < 0 || index >= grid->n_params) return -1;

    ParamSpec *spec = &grid->params[index];
    spec->name = name;
    spec->type = PARAM_INT;
    spec->data.int_vals.values = malloc(n_values * sizeof(int));
    if (!spec->data.int_vals.values) return -1;

    memcpy(spec->data.int_vals.values, values, n_values * sizeof(int));
    spec->data.int_vals.n_values = n_values;

    return 0;
}

int param_grid_add_double(ParameterGrid *grid, int index, const char *name,
                          const double *values, int n_values) {
    if (!grid || index < 0 || index >= grid->n_params) return -1;

    ParamSpec *spec = &grid->params[index];
    spec->name = name;
    spec->type = PARAM_DOUBLE;
    spec->data.double_vals.values = malloc(n_values * sizeof(double));
    if (!spec->data.double_vals.values) return -1;

    memcpy(spec->data.double_vals.values, values, n_values * sizeof(double));
    spec->data.double_vals.n_values = n_values;

    return 0;
}

int param_grid_get_n_combinations(const ParameterGrid *grid) {
    if (!grid || grid->n_params == 0) return 0;

    int n_combos = 1;
    for (int i = 0; i < grid->n_params; i++) {
        ParamSpec *spec = &grid->params[i];
        switch (spec->type) {
            case PARAM_INT:
                n_combos *= spec->data.int_vals.n_values;
                break;
            case PARAM_DOUBLE:
                n_combos *= spec->data.double_vals.n_values;
                break;
            case PARAM_ENUM:
                n_combos *= spec->data.enum_vals.n_values;
                break;
        }
    }
    return n_combos;
}

double* param_grid_get_combination(const ParameterGrid *grid, int combo_index) {
    if (!grid) return NULL;

    double *params = malloc(grid->n_params * sizeof(double));
    if (!params) return NULL;

    int divisor = 1;
    for (int i = grid->n_params - 1; i >= 0; i--) {
        ParamSpec *spec = &grid->params[i];
        int n_values = 0;

        switch (spec->type) {
            case PARAM_INT:
                n_values = spec->data.int_vals.n_values;
                break;
            case PARAM_DOUBLE:
                n_values = spec->data.double_vals.n_values;
                break;
            case PARAM_ENUM:
                n_values = spec->data.enum_vals.n_values;
                break;
        }

        int idx = (combo_index / divisor) % n_values;
        divisor *= n_values;

        switch (spec->type) {
            case PARAM_INT:
                params[i] = (double)spec->data.int_vals.values[idx];
                break;
            case PARAM_DOUBLE:
                params[i] = spec->data.double_vals.values[idx];
                break;
            case PARAM_ENUM:
                params[i] = (double)spec->data.enum_vals.values[idx];
                break;
        }
    }

    return params;
}

void param_grid_free(ParameterGrid *grid) {
    if (!grid) return;

    for (int i = 0; i < grid->n_params; i++) {
        ParamSpec *spec = &grid->params[i];
        switch (spec->type) {
            case PARAM_INT:
                free(spec->data.int_vals.values);
                break;
            case PARAM_DOUBLE:
                free(spec->data.double_vals.values);
                break;
            case PARAM_ENUM:
                free(spec->data.enum_vals.values);
                break;
        }
    }
    free(grid->params);
    free(grid);
}

/* ============================================
 * GridSearchCV Implementation
 * ============================================ */

GridSearchCV* grid_search_cv_create(
    Estimator *estimator,
    ParameterGrid *param_grid,
    int cv
) {
    GridSearchCV *gs = calloc(1, sizeof(GridSearchCV));
    if (!gs) return NULL;

    gs->base_estimator = estimator;
    gs->param_grid = param_grid;
    gs->cv = cv;
    gs->shuffle = 1;
    gs->seed = 42;
    gs->verbose = 1;
    gs->refit = 1;

    gs->is_classification = (estimator->task == TASK_CLASSIFICATION);

    int n_combos = param_grid_get_n_combinations(param_grid);
    gs->results = calloc(n_combos, sizeof(GridSearchResult));
    gs->n_results = n_combos;
    gs->best_index = -1;
    gs->best_score = -INFINITY;
    gs->best_estimator = NULL;
    gs->best_params = NULL;

    return gs;
}

// Helper: Apply parameter to LinearRegression model
static void apply_linreg_params(LinearRegression *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "learning_rate") == 0) {
            model->learning_rate = value;
        } else if (strcmp(name, "max_iter") == 0) {
            model->max_iter = (int)value;
        } else if (strcmp(name, "tol") == 0) {
            model->tol = value;
        } else if (strcmp(name, "solver") == 0) {
            model->solver = (LinRegSolver)(int)value;
        }
    }
}

// Helper: Apply parameter to DecisionTreeClassifier model
static void apply_dt_params(DecisionTreeClassifier *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "max_depth") == 0) {
            model->max_depth = (int)value;
        } else if (strcmp(name, "min_samples_split") == 0) {
            model->min_samples_split = (int)value;
        } else if (strcmp(name, "min_samples_leaf") == 0) {
            model->min_samples_leaf = (int)value;
        } else if (strcmp(name, "min_impurity_decrease") == 0) {
            model->min_impurity_decrease = value;
        } else if (strcmp(name, "max_features") == 0) {
            model->max_features = (int)value;
        }
    }
}

// Helper: Apply parameter to NeuralNetwork model
static void apply_nn_params(NeuralNetwork *model, const ParameterGrid *grid, const double *params) {
    for (int i = 0; i < grid->n_params; i++) {
        const char *name = grid->params[i].name;
        double value = params[i];

        if (strcmp(name, "learning_rate") == 0) {
            model->learning_rate = value;
        } else if (strcmp(name, "max_epochs") == 0) {
            model->max_epochs = (int)value;
        } else if (strcmp(name, "batch_size") == 0) {
            model->batch_size = (int)value;
        } else if (strcmp(name, "momentum") == 0) {
            model->momentum = value;
        } else if (strcmp(name, "l2_reg") == 0) {
            model->l2_reg = value;
        } else if (strcmp(name, "dropout_rate") == 0) {
            model->dropout_rate = value;
        }
    }
}

// Helper: Apply parameters to KNN model (k is set at create time, not tunable via GridSearch)
// KNN uses the legacy KNNModel which is not an Estimator subclass.
// If a KNNClassifier estimator wrapper is added later, this can be implemented.

// Generic: Apply parameters to any supported model type
static void apply_params(Estimator *model, ModelType type, const ParameterGrid *grid, const double *params) {
    switch (type) {
        case MODEL_LINEAR_REGRESSION:
            apply_linreg_params((LinearRegression*)model, grid, params);
            break;
        case MODEL_DECISION_TREE:
            apply_dt_params((DecisionTreeClassifier*)model, grid, params);
            break;
        case MODEL_NEURAL_NETWORK:
            apply_nn_params((NeuralNetwork*)model, grid, params);
            break;
        case MODEL_KNN:
            /* KNN uses legacy KNNModel, not tunable via GridSearch yet */
            break;
        default:
            /* Unknown model type — parameters silently ignored */
            break;
    }
}

int grid_search_cv_fit(GridSearchCV *gs, const Matrix *X, const Matrix *y) {
    if (!gs || !X || !y) return -1;

    int n_combos = gs->n_results;

    if (gs->verbose) {
        printf("Fitting %d folds for each of %d candidates, totalling %d fits\n",
               gs->cv, n_combos, gs->cv * n_combos);
    }

    clock_t total_start = clock();

    for (int combo = 0; combo < n_combos; combo++) {
        double *params = param_grid_get_combination(gs->param_grid, combo);
        if (!params) continue;

        // Clone and configure estimator
        Estimator *model = gs->base_estimator->clone(gs->base_estimator);
        if (!model) {
            free(params);
            continue;
        }

        // Apply parameters to any supported model type
        apply_params(model, gs->base_estimator->type, gs->param_grid, params);

        // Cross-validate
        clock_t fit_start = clock();
        CrossValResults *cv = cross_val_score(model, X, y, gs->cv, gs->shuffle, gs->seed);
        clock_t fit_end = clock();

        if (cv) {
            gs->results[combo].param_values = params;
            gs->results[combo].mean_test_score = cv->mean_test_score;
            gs->results[combo].std_test_score = cv->std_test_score;
            gs->results[combo].mean_train_score = cv->mean_train_score;
            gs->results[combo].mean_fit_time = (double)(fit_end - fit_start) / CLOCKS_PER_SEC;

            if (cv->mean_test_score > gs->best_score) {
                gs->best_score = cv->mean_test_score;
                gs->best_index = combo;
                free(gs->best_params);
                gs->best_params = malloc(gs->param_grid->n_params * sizeof(double));
                memcpy(gs->best_params, params, gs->param_grid->n_params * sizeof(double));
            }

            cross_val_results_free(cv);
        } else {
            free(params);
        }

        model->free(model);

        if (gs->verbose >= 2) {
            printf("[%d/%d] Score: %.6f\n", combo + 1, n_combos, gs->results[combo].mean_test_score);
        }
    }

    // Rank results
    for (int i = 0; i < n_combos; i++) {
        int rank = 1;
        for (int j = 0; j < n_combos; j++) {
            if (gs->results[j].mean_test_score > gs->results[i].mean_test_score) {
                rank++;
            }
        }
        gs->results[i].rank = rank;
    }

    // Refit best model on all data
    if (gs->refit && gs->best_index >= 0) {
        gs->best_estimator = gs->base_estimator->clone(gs->base_estimator);
        if (gs->best_estimator) {
            apply_params(gs->best_estimator, gs->base_estimator->type, gs->param_grid, gs->best_params);
        }
        if (gs->best_estimator) {
            gs->best_estimator->fit(gs->best_estimator, X, y);
        }
    }

    clock_t total_end = clock();

    if (gs->verbose) {
        printf("\nBest score: %.6f\n", gs->best_score);
        printf("Best parameters:\n");
        for (int i = 0; i < gs->param_grid->n_params; i++) {
            printf("  %s: %.6f\n", gs->param_grid->params[i].name, gs->best_params[i]);
        }
        printf("Total time: %.2f seconds\n", (double)(total_end - total_start) / CLOCKS_PER_SEC);
    }

    return 0;
}

const double* grid_search_cv_best_params(const GridSearchCV *gs) {
    return gs ? gs->best_params : NULL;
}

double grid_search_cv_best_score(const GridSearchCV *gs) {
    return gs ? gs->best_score : -1.0;
}

Estimator* grid_search_cv_best_estimator(GridSearchCV *gs) {
    return gs ? gs->best_estimator : NULL;
}

Matrix* grid_search_cv_predict(const GridSearchCV *gs, const Matrix *X) {
    if (!gs || !gs->best_estimator) return NULL;
    return gs->best_estimator->predict(gs->best_estimator, X);
}

double grid_search_cv_score(const GridSearchCV *gs, const Matrix *X, const Matrix *y) {
    if (!gs || !gs->best_estimator) return -1.0;
    return gs->best_estimator->score(gs->best_estimator, X, y);
}

void grid_search_cv_print_results(const GridSearchCV *gs) {
    if (!gs) return;

    printf("\n=== GridSearchCV Results ===\n\n");

    // Print header
    printf("%-6s ", "Rank");
    for (int i = 0; i < gs->param_grid->n_params; i++) {
        printf("%-15s ", gs->param_grid->params[i].name);
    }
    printf("%-15s %-15s\n", "Mean Score", "Std Score");
    printf("--------------------------------------------------------------\n");

    // Print results sorted by rank
    for (int rank = 1; rank <= gs->n_results; rank++) {
        for (int i = 0; i < gs->n_results; i++) {
            if (gs->results[i].rank == rank && gs->results[i].param_values) {
                printf("%-6d ", rank);
                for (int j = 0; j < gs->param_grid->n_params; j++) {
                    printf("%-15.6f ", gs->results[i].param_values[j]);
                }
                printf("%-15.6f %-15.6f\n",
                       gs->results[i].mean_test_score,
                       gs->results[i].std_test_score);
                break;  // Only print first with this rank
            }
        }
        if (rank >= 10) {
            printf("... (%d more)\n", gs->n_results - 10);
            break;
        }
    }

    printf("\nBest: rank=%d, score=%.6f\n", gs->results[gs->best_index].rank, gs->best_score);
    printf("============================\n\n");
}

void grid_search_cv_free(GridSearchCV *gs) {
    if (!gs) return;

    for (int i = 0; i < gs->n_results; i++) {
        free(gs->results[i].param_values);
    }
    free(gs->results);
    free(gs->best_params);

    if (gs->best_estimator) {
        gs->best_estimator->free(gs->best_estimator);
    }

    free(gs);
}

/* ============================================
 * Learning Curves Implementation
 * ============================================ */

LearningCurveResult* learning_curve(
    const Estimator *estimator,
    const Matrix *X,
    const Matrix *y,
    const double *train_sizes,
    int n_sizes,
    int cv
) {
    if (!estimator || !X || !y || !train_sizes || n_sizes <= 0) return NULL;

    LearningCurveResult *lc = calloc(1, sizeof(LearningCurveResult));
    if (!lc) return NULL;

    lc->n_points = n_sizes;
    lc->train_sizes = malloc(n_sizes * sizeof(size_t));
    lc->train_scores_mean = malloc(n_sizes * sizeof(double));
    lc->train_scores_std = malloc(n_sizes * sizeof(double));
    lc->test_scores_mean = malloc(n_sizes * sizeof(double));
    lc->test_scores_std = malloc(n_sizes * sizeof(double));

    if (!lc->train_sizes || !lc->train_scores_mean || !lc->train_scores_std ||
        !lc->test_scores_mean || !lc->test_scores_std) {
        learning_curve_free(lc);
        return NULL;
    }

    size_t n_samples = X->rows;

    for (int s = 0; s < n_sizes; s++) {
        // Convert fraction to absolute size
        size_t size;
        if (train_sizes[s] <= 1.0) {
            size = (size_t)(train_sizes[s] * n_samples);
        } else {
            size = (size_t)train_sizes[s];
        }
        if (size < 1) size = 1;
        if (size > n_samples) size = n_samples;
        lc->train_sizes[s] = size;

        // Create subset
        Matrix *X_sub = matrix_alloc(size, X->cols);
        Matrix *y_sub = matrix_alloc(size, y->cols);

        // Random sample
        size_t *indices = malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) indices[i] = i;
        shuffle_indices(indices, n_samples);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < X->cols; j++) {
                X_sub->data[i * X->cols + j] = X->data[indices[i] * X->cols + j];
            }
            for (size_t j = 0; j < y->cols; j++) {
                y_sub->data[i * y->cols + j] = y->data[indices[i] * y->cols + j];
            }
        }
        free(indices);

        // Cross-validate on subset
        CrossValResults *cv_result = cross_val_score(estimator, X_sub, y_sub, cv, 1, 42);

        if (cv_result) {
            lc->train_scores_mean[s] = cv_result->mean_train_score;
            lc->train_scores_std[s] = cv_result->std_train_score;
            lc->test_scores_mean[s] = cv_result->mean_test_score;
            lc->test_scores_std[s] = cv_result->std_test_score;
            cross_val_results_free(cv_result);
        }

        matrix_free(X_sub);
        matrix_free(y_sub);
    }

    return lc;
}

void learning_curve_print(const LearningCurveResult *lc) {
    if (!lc) return;

    printf("\n=== Learning Curve ===\n");
    printf("%-12s %-15s %-15s %-15s %-15s\n",
           "Train Size", "Train Mean", "Train Std", "Test Mean", "Test Std");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < lc->n_points; i++) {
        printf("%-12zu %-15.6f %-15.6f %-15.6f %-15.6f\n",
               lc->train_sizes[i],
               lc->train_scores_mean[i], lc->train_scores_std[i],
               lc->test_scores_mean[i], lc->test_scores_std[i]);
    }
    printf("======================\n\n");
}

int learning_curve_save_csv(const LearningCurveResult *lc, const char *filename) {
    if (!lc || !filename) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "train_size,train_score_mean,train_score_std,test_score_mean,test_score_std\n");
    for (int i = 0; i < lc->n_points; i++) {
        fprintf(f, "%zu,%.10f,%.10f,%.10f,%.10f\n",
                lc->train_sizes[i],
                lc->train_scores_mean[i], lc->train_scores_std[i],
                lc->test_scores_mean[i], lc->test_scores_std[i]);
    }

    fclose(f);
    return 0;
}

void learning_curve_free(LearningCurveResult *lc) {
    if (!lc) return;
    free(lc->train_sizes);
    free(lc->train_scores_mean);
    free(lc->train_scores_std);
    free(lc->test_scores_mean);
    free(lc->test_scores_std);
    free(lc);
}


/* --- pipeline.c --- */
/**
 * pipeline.c - Pipeline implementation
 */




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



#endif /* CML_IMPLEMENTATION */
#endif /* TINYCML_H */
