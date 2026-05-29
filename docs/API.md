# API Reference

Complete API documentation for tinycml.

## Table of Contents

1. [Unified Estimator API](#unified-estimator-api)
2. [Matrix Operations](#matrix-operations)
3. [Vector Operations](#vector-operations)
4. [Utility Functions](#utility-functions)
5. [CSV Handling](#csv-handling)
6. [Data Preprocessing](#data-preprocessing)
7. [Pipeline](#pipeline)
8. [Cross-Validation](#cross-validation)
9. [Model Selection](#model-selection)
10. [Linear Regression](#linear-regression)
11. [Logistic Regression](#logistic-regression)
12. [k-Nearest Neighbors](#k-nearest-neighbors)
13. [Naive Bayes](#naive-bayes)
14. [Decision Tree](#decision-tree)
15. [Random Forest](#random-forest)
16. [Neural Network](#neural-network)
17. [k-Means Clustering](#k-means-clustering)
18. [PCA](#pca)
19. [Feature Selection](#feature-selection)
20. [Evaluation Metrics](#evaluation-metrics)

---

## Unified Estimator API

**Header:** `estimator.h`

All models in tinycml follow a consistent interface inspired by scikit-learn.

### Base Structure

```c
typedef struct Estimator {
    ModelType type;
    TaskType task;
    int is_fitted;

    // Virtual function table
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
} Estimator;
```

### Model Types

```c
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
```

### Task Types

```c
typedef enum {
    TASK_REGRESSION,
    TASK_CLASSIFICATION,
    TASK_CLUSTERING,
    TASK_TRANSFORMATION
} TaskType;
```

### Verbose Levels

```c
typedef enum {
    VERBOSE_SILENT = 0,    // No output
    VERBOSE_MINIMAL = 1,   // Only final result
    VERBOSE_PROGRESS = 2,  // Progress bar / periodic updates
    VERBOSE_DETAILED = 3   // Every epoch
} VerboseLevel;
```

### Training History

```c
typedef struct {
    double *loss_history;
    double *metric_history;
    size_t n_epochs;
    size_t capacity;
    int converged;
    size_t best_epoch;
} TrainingHistory;

TrainingHistory* training_history_alloc(size_t initial_capacity);
void training_history_append(TrainingHistory *history, double loss, double metric);
void training_history_free(TrainingHistory *history);
int training_history_save_csv(const TrainingHistory *history, const char *filename);
```

### Utility Functions

```c
int estimator_check_fitted(const Estimator *est, const char *method_name);
void estimator_set_verbose(Estimator *est, VerboseLevel level);
void estimator_set_callback(Estimator *est, TrainingCallback cb, void *user_data);
const TrainingHistory* estimator_get_history(const Estimator *est);
Estimator* estimator_clone(const Estimator *est);
```

### Usage Example

```c
#include "estimator.h"
#include "linear_regression.h"

// Create model
LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);

// Enable verbose output
model->base.verbose = VERBOSE_PROGRESS;

// Train
model->base.fit((Estimator*)model, X_train, y_train);

// Predict
Matrix *predictions = model->base.predict((Estimator*)model, X_test);

// Evaluate
double r2 = model->base.score((Estimator*)model, X_test, y_test);

// Save model
model->base.save((Estimator*)model, "model.bin");

// Free
model->base.free((Estimator*)model);
```

---

## Matrix Operations

**Header:** `matrix.h`

### Data Structure

```c
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of columns
    double *data;   // Row-major data array
} Matrix;
```

**Memory Layout:** Row-major order. Element (i, j) is at `data[i * cols + j]`.

### Memory Management

#### `matrix_alloc`
```c
Matrix* matrix_alloc(size_t rows, size_t cols);
```
Allocate a zero-initialized matrix.

#### `matrix_free`
```c
void matrix_free(Matrix *m);
```
Free matrix memory. Safe to call with `NULL`.

#### `matrix_copy`
```c
Matrix* matrix_copy(const Matrix *m);
```
Create a deep copy of a matrix.

### Element Access

#### `matrix_get`
```c
double matrix_get(const Matrix *m, size_t i, size_t j);
```
Get element at position (i, j).

#### `matrix_set`
```c
void matrix_set(Matrix *m, size_t i, size_t j, double val);
```
Set element at position (i, j).

### Arithmetic Operations

```c
Matrix* matrix_add(const Matrix *a, const Matrix *b);      // Element-wise addition
Matrix* matrix_sub(const Matrix *a, const Matrix *b);      // Element-wise subtraction
Matrix* matrix_mul(const Matrix *a, const Matrix *b);      // Element-wise multiplication
Matrix* matrix_scale(const Matrix *m, double scalar);       // Scalar multiplication
Matrix* matrix_matmul(const Matrix *a, const Matrix *b);   // Matrix multiplication
Matrix* matrix_transpose(const Matrix *m);                  // Transpose
```

### Row/Column Operations

```c
Matrix* matrix_get_rows(const Matrix *m, const size_t *indices, size_t n);
Matrix* matrix_get_cols(const Matrix *m, const size_t *indices, size_t n);
```

### Utilities

```c
void matrix_print(const Matrix *m);
void matrix_fill(Matrix *m, double val);
Matrix* matrix_identity(size_t n);
```

---

## Vector Operations

**Header:** `vector.h`

Vectors are represented as (n×1) or (1×n) matrices.

```c
double vector_dot(const Matrix *a, const Matrix *b);   // Dot product
double vector_norm(const Matrix *v);                    // L2 norm
Matrix* vector_scale(const Matrix *v, double scalar);  // Scalar multiplication
Matrix* vector_add(const Matrix *a, const Matrix *b);  // Addition
Matrix* vector_sub(const Matrix *a, const Matrix *b);  // Subtraction
```

---

## Utility Functions

**Header:** `utils.h`

### Random Number Generation

```c
void rand_seed(unsigned int seed);
double rand_uniform(void);                              // [0, 1)
double rand_uniform_range(double min, double max);      // [min, max)
double rand_normal(void);                               // N(0, 1)
double rand_normal_params(double mean, double std);     // N(mean, std)
```

### Statistics

```c
double mean(const double *data, size_t n);
double std_dev(const double *data, size_t n);
double variance(const double *data, size_t n);
void shuffle_indices(size_t *indices, size_t n);
```

---

## CSV Handling

**Header:** `csv.h`

```c
Matrix* csv_load(const char *filename, int has_header);
int csv_save(const Matrix *m, const char *filename);
```

---

## Data Preprocessing

**Header:** `preprocessing.h`

### Train/Test Split

```c
typedef struct {
    Matrix *X_train;
    Matrix *X_test;
    Matrix *y_train;
    Matrix *y_test;
} TrainTestSplit;

TrainTestSplit train_test_split(const Matrix *X, const Matrix *y,
                                 double test_ratio, unsigned int seed);
void train_test_split_free(TrainTestSplit *split);
```

### StandardScaler

```c
StandardScaler* standard_scaler_create(void);
// Uses estimator API: fit, transform, fit_transform
```

### MinMaxScaler

```c
MinMaxScaler* minmax_scaler_create(void);
// Uses estimator API: fit, transform
```

### OneHotEncoder

```c
OneHotEncoder* one_hot_encoder_create(void);
// Uses estimator API: fit, transform
```

### PolynomialFeatures

```c
PolynomialFeatures* polynomial_features_create(int degree, int include_bias);
// Uses estimator API: fit, transform
```

### Bias Column

```c
Matrix* add_bias_column(const Matrix *X);
```

---

## Pipeline

**Header:** `pipeline.h`

Chain preprocessing steps with models.

```c
typedef struct {
    Estimator base;
    char **step_names;
    Estimator **steps;
    int n_steps;
} Pipeline;

Pipeline* pipeline_create(void);
void pipeline_add_step(Pipeline *pipe, const char *name, Estimator *step);
Estimator* pipeline_get_step(Pipeline *pipe, const char *name);
void pipeline_free(Pipeline *pipe);
```

### Usage

```c
Pipeline *pipe = pipeline_create();
pipeline_add_step(pipe, "scaler", (Estimator*)standard_scaler_create());
pipeline_add_step(pipe, "model", (Estimator*)linear_regression_create(LINREG_SOLVER_CLOSED));

pipe->base.fit((Estimator*)pipe, X_train, y_train);
Matrix *pred = pipe->base.predict((Estimator*)pipe, X_test);

pipeline_free(pipe);
```

---

## Cross-Validation

**Header:** `validation.h`

### K-Fold

```c
typedef struct {
    size_t **train_indices;
    size_t **test_indices;
    size_t *train_sizes;
    size_t *test_sizes;
    int n_splits;
} KFold;

KFold* kfold_create(size_t n_samples, int n_splits, int shuffle, unsigned int seed);
void kfold_free(KFold *kf);
```

### Cross-Validation Score

```c
typedef struct {
    double *train_scores;
    double *test_scores;
    double mean_train_score;
    double mean_test_score;
    double std_train_score;
    double std_test_score;
    int n_splits;
} CrossValResults;

CrossValResults* cross_val_score(Estimator *estimator, const Matrix *X,
                                  const Matrix *y, int cv, int shuffle,
                                  unsigned int seed);
void cross_val_results_free(CrossValResults *results);
```

---

## Model Selection

**Header:** `model_selection.h`

### Parameter Grid

```c
typedef struct {
    char **param_names;
    int *param_types;      // 0=int, 1=double
    void **param_values;
    int *param_counts;
    int n_params;
} ParamGrid;

void param_grid_init(ParamGrid *grid);
void param_grid_add_int(ParamGrid *grid, const char *name, int *values, int count);
void param_grid_add_double(ParamGrid *grid, const char *name, double *values, int count);
void param_grid_free(ParamGrid *grid);
```

### GridSearchCV

```c
typedef struct {
    Estimator base;
    Estimator *estimator;
    ParamGrid *param_grid;
    int cv;
    double best_score_;
    int *best_params_idx;
    Estimator *best_estimator_;
} GridSearchCV;

GridSearchCV* grid_search_cv_create(Estimator *estimator, ParamGrid *grid,
                                     int cv, unsigned int seed);
int grid_search_get_best_int(GridSearchCV *gs, const char *param_name);
double grid_search_get_best_double(GridSearchCV *gs, const char *param_name);
void grid_search_cv_free(GridSearchCV *gs);
```

### Usage

```c
ParamGrid grid;
param_grid_init(&grid);
param_grid_add_int(&grid, "max_depth", (int[]){3, 5, 10}, 3);

GridSearchCV *gs = grid_search_cv_create((Estimator*)dt, &grid, 5, 42);
gs->base.fit((Estimator*)gs, X, y);

printf("Best score: %.4f\n", gs->best_score_);
printf("Best max_depth: %d\n", grid_search_get_best_int(gs, "max_depth"));

grid_search_cv_free(gs);
param_grid_free(&grid);
```

---

## Linear Regression

**Header:** `linear_regression.h`

```c
typedef enum {
    LINREG_SOLVER_CLOSED,
    LINREG_SOLVER_GD
} LinRegSolver;

typedef struct {
    Estimator base;
    LinRegSolver solver;
    double learning_rate;
    int epochs;
    Matrix *weights_;
} LinearRegression;

LinearRegression* linear_regression_create(LinRegSolver solver);
LinearRegression* linear_regression_create_full(LinRegSolver solver,
                                                 double lr, int epochs);
```

---

## Logistic Regression

**Header:** `logistic_regression.h`

```c
typedef struct {
    Estimator base;
    double learning_rate;
    int epochs;
    double l2_reg;
    Matrix *weights_;
} LogisticRegression;

LogisticRegression* logistic_regression_create(void);
LogisticRegression* logistic_regression_create_full(double lr, int epochs, double l2);
```

---

## k-Nearest Neighbors

**Header:** `knn.h`

```c
typedef struct {
    Estimator base;
    int k;
    Matrix *X_train_;
    Matrix *y_train_;
} KNNClassifier;

KNNClassifier* knn_classifier_create(int k);
```

---

## Naive Bayes

**Header:** `naive_bayes.h`

```c
typedef struct {
    Estimator base;
    double *class_priors_;
    double *means_;
    double *variances_;
    int n_classes_;
    int n_features_;
} GaussianNB;

GaussianNB* gaussian_nb_create(void);
```

---

## Decision Tree

**Header:** `decision_tree.h`

```c
typedef enum {
    CRITERION_GINI,
    CRITERION_ENTROPY
} SplitCriterion;

typedef struct {
    Estimator base;
    SplitCriterion criterion;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    double min_impurity_decrease;
    struct TreeNode *root_;
} DecisionTreeClassifier;

DecisionTreeClassifier* decision_tree_classifier_create(void);
DecisionTreeClassifier* decision_tree_classifier_create_full(
    SplitCriterion criterion, int max_depth, int min_samples_split,
    int min_samples_leaf, double min_impurity_decrease);
```

---

## Random Forest

**Header:** `ensemble.h`

```c
typedef struct {
    Estimator base;
    int n_estimators;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;
    int bootstrap;
    unsigned int seed;
    DecisionTreeClassifier **trees_;
    double oob_score_;
} RandomForestClassifier;

RandomForestClassifier* random_forest_classifier_create(int n_estimators);
RandomForestClassifier* random_forest_classifier_create_full(
    int n_estimators, int max_depth, int min_samples_split,
    int min_samples_leaf, int max_features, int bootstrap, unsigned int seed);
```

### Usage

```c
RandomForestClassifier *rf = random_forest_classifier_create_full(
    100,    // n_estimators
    10,     // max_depth
    2,      // min_samples_split
    1,      // min_samples_leaf
    0,      // max_features (0 = sqrt)
    1,      // bootstrap
    42      // seed
);

rf->base.fit((Estimator*)rf, X_train, y_train);
printf("OOB score: %.4f\n", rf->oob_score_);

Matrix *proba = rf->base.predict_proba((Estimator*)rf, X_test);
```

---

## Neural Network

**Header:** `neural_network.h`

```c
typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
} ActivationType;

typedef struct {
    Estimator base;
    size_t *layer_sizes;
    int n_layers;
    ActivationType hidden_activation;
    double learning_rate;
    int epochs;
    int batch_size;
    Matrix **weights_;
    Matrix **biases_;
} NeuralNetwork;

NeuralNetwork* neural_network_create(size_t *layer_sizes, int n_layers,
                                      ActivationType hidden_activation);
```

### Usage

```c
size_t layers[] = {n_features, 64, 32, n_classes};
NeuralNetwork *nn = neural_network_create(layers, 4, ACTIVATION_RELU);

nn->learning_rate = 0.001;
nn->epochs = 100;
nn->batch_size = 32;

nn->base.fit((Estimator*)nn, X_train, y_train);
```

---

## k-Means Clustering

**Header:** `kmeans.h`

```c
typedef struct {
    Estimator base;
    int k;
    int max_iter;
    unsigned int seed;
    Matrix *centroids_;
    double inertia_;
} KMeans;

KMeans* kmeans_create(int k);
KMeans* kmeans_create_full(int k, int max_iter, unsigned int seed);
```

---

## PCA

**Header:** `decomposition.h`

```c
typedef struct {
    Estimator base;
    int n_components;
    int whiten;
    double *explained_variance_;
    double *explained_variance_ratio_;
    Matrix *components_;
    double *mean_;
} PCA;

PCA* pca_create(int n_components);
PCA* pca_create_full(int n_components, int whiten);
const double* pca_explained_variance_ratio(const PCA *pca);
double pca_cumulative_variance(const PCA *pca, int n_components);
Matrix* pca_inverse_transform(const PCA *pca, const Matrix *X_transformed);
```

### Usage

```c
PCA *pca = pca_create(2);
pca->base.fit((Estimator*)pca, X, NULL);

Matrix *X_reduced = pca->base.transform((Estimator*)pca, X);

const double *evr = pca_explained_variance_ratio(pca);
printf("Variance explained: %.2f%%\n", (evr[0] + evr[1]) * 100);

Matrix *X_reconstructed = pca_inverse_transform(pca, X_reduced);
```

---

## Feature Selection

**Header:** `feature_selection.h`

### Scoring Functions

```c
typedef enum {
    SCORE_F_CLASSIF,
    SCORE_F_REGRESSION,
    SCORE_MUTUAL_INFO_CLASSIF,
    SCORE_MUTUAL_INFO_REGRESSION,
    SCORE_CHI2
} ScoreFunction;

void f_classif(const Matrix *X, const Matrix *y, double *f_scores, double *p_values);
void f_regression(const Matrix *X, const Matrix *y, double *f_scores, double *p_values);
void chi2(const Matrix *X, const Matrix *y, double *chi2_scores, double *p_values);
void mutual_info_classif(const Matrix *X, const Matrix *y, double *mi_scores, int n_neighbors);
void mutual_info_regression(const Matrix *X, const Matrix *y, double *mi_scores, int n_neighbors);
```

### SelectKBest

```c
typedef struct {
    Estimator base;
    ScoreFunction score_func;
    int k;
    double *scores_;
    double *pvalues_;
    int *support_;
    int n_features_;
    int n_features_selected_;
} SelectKBest;

SelectKBest* select_k_best_create(ScoreFunction score_func, int k);
const double* select_k_best_get_scores(const SelectKBest *skb);
const int* select_k_best_get_support(const SelectKBest *skb);
```

### VarianceThreshold

```c
typedef struct {
    Estimator base;
    double threshold;
    double *variances_;
    int *support_;
    int n_features_;
    int n_features_selected_;
} VarianceThreshold;

VarianceThreshold* variance_threshold_create(double threshold);
const double* variance_threshold_variances(const VarianceThreshold *vt);
const int* variance_threshold_get_support(const VarianceThreshold *vt);
```

### Usage

```c
// SelectKBest
SelectKBest *skb = select_k_best_create(SCORE_F_REGRESSION, 5);
skb->base.fit((Estimator*)skb, X, y);
Matrix *X_selected = skb->base.transform((Estimator*)skb, X);

const int *support = select_k_best_get_support(skb);
for (int j = 0; j < skb->n_features_; j++) {
    if (support[j]) printf("Feature %d selected\n", j);
}

// VarianceThreshold
VarianceThreshold *vt = variance_threshold_create(0.1);
vt->base.fit((Estimator*)vt, X, NULL);
Matrix *X_filtered = vt->base.transform((Estimator*)vt, X);
```

---

## Evaluation Metrics

**Header:** `metrics.h`

### Regression Metrics

```c
double mse(const Matrix *y_true, const Matrix *y_pred);    // Mean Squared Error
double rmse(const Matrix *y_true, const Matrix *y_pred);   // Root MSE
double mae(const Matrix *y_true, const Matrix *y_pred);    // Mean Absolute Error
double r2_score(const Matrix *y_true, const Matrix *y_pred); // R-squared
```

### Classification Metrics

```c
double accuracy(const Matrix *y_true, const Matrix *y_pred);
double precision(const Matrix *y_true, const Matrix *y_pred);
double recall(const Matrix *y_true, const Matrix *y_pred);
double f1_score(const Matrix *y_true, const Matrix *y_pred);
```

### Confusion Matrix

```c
typedef struct {
    int tp;  // True Positives
    int tn;  // True Negatives
    int fp;  // False Positives
    int fn;  // False Negatives
} ConfusionMatrix;

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred);
void confusion_matrix_print(const ConfusionMatrix *cm);
```

### Clustering Metrics

```c
double silhouette_score(const Matrix *X, const Matrix *labels);
```
