# TinyCML Model Zoo

Pre-trained models for common datasets. Load directly with the C API or Python bindings.

## Available Models

| Model | Dataset | Accuracy | File |
|-------|---------|----------|------|
| Logistic Regression | Iris (binary) | ~97% | `iris_logreg.bin` |
| KNN (k=5) | Iris (binary) | ~100% | `iris_knn.bin` |
| Gaussian NB | Iris (binary) | ~100% | `iris_nb.bin` |
| Decision Tree | Iris (binary) | ~100% | `iris_dtree.bin` |
| Linear Regression | Diabetes | R²~0.50 | `diabetes_linreg.bin` |
| Random Forest | Iris (binary) | ~100% | `iris_rf.bin` |
| Gradient Boosting | Diabetes | R²~0.60 | `diabetes_gb.bin` |

## Usage (C)

```c
#include "logistic_regression.h"

Estimator *model = logistic_regression_load("models/iris_logreg.bin");
Matrix *pred = model->base.predict(model, X);
```

## Usage (Python)

```python
from tinycml import LogisticRegression
model = LogisticRegression().load("models/iris_logreg.bin")
predictions = model.predict(X)
```

## Regeneration

```bash
python3 scripts/generate_zoo.py
```
