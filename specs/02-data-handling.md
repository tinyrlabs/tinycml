# Data Handling Specification

## CSV Loader

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `csv_load` | `Matrix* csv_load(const char *filename, int has_header)` | Load CSV to Matrix |
| `csv_save` | `int csv_save(const Matrix *m, const char *filename)` | Save Matrix to CSV |

### CSV Format

- Comma-separated values
- Numeric data only (double)
- Optional header row (skipped if `has_header=1`)
- Unix or Windows line endings

### Error Handling

- Return `NULL` if file not found
- Return `NULL` if parse error
- Print error message to stderr

---

## Data Preprocessing

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

- `test_ratio`: fraction for test set (e.g., 0.2 = 20%)
- `seed`: random seed for reproducibility
- Shuffles data before splitting

### Standardization (Z-Score)

```c
typedef struct {
    double *means;
    double *stds;
    size_t n_features;
} Scaler;

Scaler* standardize_fit(const Matrix *X);
Matrix* standardize_transform(const Matrix *X, const Scaler *scaler);
Matrix* standardize_fit_transform(const Matrix *X, Scaler **scaler_out);
void scaler_free(Scaler *scaler);
```

Formula: `z = (x - mean) / std`

### Min-Max Scaling

```c
typedef struct {
    double *mins;
    double *maxs;
    size_t n_features;
} MinMaxScaler;

MinMaxScaler* minmax_fit(const Matrix *X);
Matrix* minmax_transform(const Matrix *X, const MinMaxScaler *scaler);
void minmax_scaler_free(MinMaxScaler *scaler);
```

Formula: `x_scaled = (x - min) / (max - min)`

---

## Sample Datasets

### simple_linear.csv

```csv
x,y
1.0,2.1
2.0,3.9
3.0,6.1
4.0,7.8
5.0,10.2
```

### binary_classification.csv

```csv
x1,x2,label
1.0,2.0,0
1.5,1.8,0
2.0,2.5,0
3.0,3.5,1
3.5,4.0,1
4.0,3.8,1
```

### clusters.csv

```csv
x1,x2
1.0,1.0
1.2,0.8
0.9,1.1
5.0,5.0
5.2,4.8
4.9,5.1
```
