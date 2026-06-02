# tinycml — Python Bindings

Pure C11 machine learning library with sklearn-compatible Python API.

## Installation

```bash
# Build the C library first
cd /path/to/tinycml
make build

# Install Python package
cd python
pip install -e .
```

Or set the library path manually:
```bash
export TINYCML_LIB=/path/to/libtinycml.so
```

## Quick Start

```python
from tinycml import LinearRegression, LogisticRegression, KNNClassifier
import numpy as np

# Regression
X = np.random.randn(100, 3)
y = X @ [1.5, -2.0, 0.5] + 0.1

model = LinearRegression()
model.fit(X, y)
print(f"R² = {model.score(X, y):.4f}")

# Classification
X = np.random.randn(200, 2)
y = (X[:, 0] > 0).astype(float)

clf = LogisticRegression()
clf.fit(X, y)
print(f"Accuracy = {clf.score(X, y):.4f}")
print(f"Proba shape = {clf.predict_proba(X).shape}")

# Save / Load
model.save("my_model.bin")
model2 = LinearRegression().load("my_model.bin")
```

## Available Models

**Regression:**
- `LinearRegression()`
- `Ridge(alpha=1.0)`
- `Lasso(alpha=1.0, max_iter=1000)`
- `SGDRegressor(alpha=0.0001, learning_rate=0.01, max_iter=1000)`
- `DecisionTreeRegressor(max_depth=10)`
- `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)`

**Classification:**
- `LogisticRegression(learning_rate=0.01, max_iter=1000)`
- `KNNClassifier(n_neighbors=5)`
- `SVMClassifier(C=1.0, max_iter=1000)`
- `GaussianNB()`
- `DecisionTreeClassifier(max_depth=10)`
- `RandomForestClassifier(n_estimators=100)`
- `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)`
- `SGDClassifier(alpha=0.0001, learning_rate=0.01)`

## API

Every model follows the sklearn interface:

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the model |
| `predict(X)` | Predict labels/values |
| `predict_proba(X)` | Class probabilities (classification only) |
| `score(X, y)` | Accuracy or R² |
| `save(path)` | Save to binary file |
| `load(path)` | Load from binary file |

## Running Tests

```bash
cd python
python tests/test_bindings.py
```
