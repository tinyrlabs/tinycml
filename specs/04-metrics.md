# Evaluation Metrics Specification

## Regression Metrics

### Mean Squared Error (MSE)

```c
double mse(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `MSE = (1/n) * sum((y_true - y_pred)^2)`

### Root Mean Squared Error (RMSE)

```c
double rmse(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `RMSE = sqrt(MSE)`

### Mean Absolute Error (MAE)

```c
double mae(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `MAE = (1/n) * sum(|y_true - y_pred|)`

---

## Classification Metrics

### Accuracy

```c
double accuracy(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `accuracy = correct_predictions / total_predictions`

### Precision

```c
double precision(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `precision = TP / (TP + FP)`

- TP = True Positives (predicted 1, actual 1)
- FP = False Positives (predicted 1, actual 0)

### Recall

```c
double recall(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `recall = TP / (TP + FN)`

- FN = False Negatives (predicted 0, actual 1)

### F1 Score

```c
double f1_score(const Matrix *y_true, const Matrix *y_pred);
```

Formula: `F1 = 2 * (precision * recall) / (precision + recall)`

---

## Confusion Matrix

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

---

## Test Cases

### MSE Test

```c
// y_true = [1, 2, 3], y_pred = [1, 2, 3]
// MSE = 0.0

// y_true = [1, 2, 3], y_pred = [2, 3, 4]
// MSE = 1.0
```

### Accuracy Test

```c
// y_true = [0, 0, 1, 1], y_pred = [0, 1, 1, 1]
// accuracy = 3/4 = 0.75
```

### Precision/Recall Test

```c
// y_true = [0, 0, 1, 1, 1], y_pred = [0, 1, 1, 0, 1]
// TP = 2, FP = 1, FN = 1, TN = 1
// precision = 2/3 ≈ 0.667
// recall = 2/3 ≈ 0.667
// F1 = 2/3 ≈ 0.667
```
