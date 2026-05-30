# ML Algorithms Specification

## Linear Regression

### Model

`y = X * w + b` (or `y = X_aug * w` with bias column)

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `linreg_fit_closed` | `Matrix* linreg_fit_closed(const Matrix *X, const Matrix *y)` | Normal equation: w = (X'X)^(-1) X'y |
| `linreg_fit_gd` | `Matrix* linreg_fit_gd(const Matrix *X, const Matrix *y, double lr, int epochs)` | Gradient descent |
| `linreg_predict` | `Matrix* linreg_predict(const Matrix *X, const Matrix *weights)` | Predict y values |

### Notes

- Add bias column (ones) internally or expect user to add it
- For closed-form, may need pseudo-inverse for stability
- Gradient descent: `w = w - lr * X' * (X*w - y) / n`

---

## Logistic Regression

### Model

Binary classification: `P(y=1|x) = sigmoid(X * w)`

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `logreg_fit` | `Matrix* logreg_fit(const Matrix *X, const Matrix *y, double lr, int epochs)` | Train with gradient descent |
| `logreg_predict_proba` | `Matrix* logreg_predict_proba(const Matrix *X, const Matrix *weights)` | Predict probabilities |
| `logreg_predict` | `Matrix* logreg_predict(const Matrix *X, const Matrix *weights, double threshold)` | Predict classes (0/1) |

### Sigmoid Function

```c
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
```

### Gradient

`gradient = X' * (sigmoid(X*w) - y) / n`

---

## k-Nearest Neighbors

### Model

Classification by majority vote of k nearest training examples.

### Data Structure

```c
typedef struct {
    Matrix *X_train;
    Matrix *y_train;
    int k;
} KNNModel;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `knn_fit` | `KNNModel* knn_fit(const Matrix *X, const Matrix *y, int k)` | Store training data |
| `knn_predict` | `Matrix* knn_predict(const KNNModel *model, const Matrix *X)` | Predict class labels |
| `knn_free` | `void knn_free(KNNModel *model)` | Free model |

### Distance

Euclidean distance: `d = sqrt(sum((x_i - y_i)^2))`

---

## k-Means Clustering

### Model

Partition n observations into k clusters.

### Data Structure

```c
typedef struct {
    Matrix *centroids;  // k × n_features
    int k;
    int max_iter;
} KMeansModel;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `kmeans_fit` | `KMeansModel* kmeans_fit(const Matrix *X, int k, int max_iter, unsigned int seed)` | Lloyd's algorithm |
| `kmeans_predict` | `Matrix* kmeans_predict(const KMeansModel *model, const Matrix *X)` | Assign cluster labels |
| `kmeans_free` | `void kmeans_free(KMeansModel *model)` | Free model |

### Algorithm (Lloyd's)

1. Initialize k centroids randomly from data points
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence or max_iter
