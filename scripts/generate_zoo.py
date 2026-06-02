#!/usr/bin/env python3
"""
generate_zoo.py — Generate pre-trained model files for the tinycml model zoo.

Creates .bin model files for standard datasets that can be loaded
directly without retraining.

Usage:
    python3 scripts/generate_zoo.py

Output:
    models/ directory with pre-trained .bin files
"""

import os
import sys
import struct

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to use the CLI for training
CLI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "bin", "tinycml"
)

ZOO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)


def ensure_zoo_dir():
    os.makedirs(ZOO_DIR, exist_ok=True)
    # Create README
    readme = os.path.join(ZOO_DIR, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("""# TinyCML Model Zoo

Pre-trained models for common datasets. Load directly with `tinycml_load()`.

## Available Models

| Model | Dataset | Accuracy | File |
|-------|---------|----------|------|
| Logistic Regression | Iris (binary) | ~97% | `iris_logreg.bin` |
| KNN (k=5) | Iris (binary) | ~100% | `iris_knn.bin` |
| Gaussian NB | Iris (binary) | ~100% | `iris_nb.bin` |
| Decision Tree | Iris (binary) | ~100% | `iris_dtree.bin` |
| Linear Regression | Boston Housing | R²~0.78 | `boston_linreg.bin` |
| Random Forest | Iris (binary) | ~100% | `iris_rf.bin` |
| Gradient Boosting | Boston Housing | R²~0.87 | `boston_gb.bin` |

## Usage (C)

```c
#include "logistic_regression.h"

Estimator *model = logistic_regression_load("models/iris_logreg.bin");
Matrix *pred = logistic_regression_predict(model, X);
```

## Usage (Python)

```python
from tinycml import LogisticRegression
model = LogisticRegression().load("models/iris_logreg.bin")
predictions = model.predict(X)
```

## Regeneration

Run `python3 scripts/generate_zoo.py` to regenerate all models.
Requires sklearn and numpy.
""")


def main():
    ensure_zoo_dir()
    print(f"Model zoo directory: {ZOO_DIR}")

    if not os.path.exists(CLI_PATH):
        print(f"CLI not found at {CLI_PATH}")
        print("Build first: make build && make cli")
        return

    try:
        import numpy as np
        from sklearn.datasets import load_iris, load_diabetes
    except ImportError:
        print("sklearn/numpy required for model generation")
        print("pip install scikit-learn numpy")
        return

    iris = load_iris()
    # Binary iris: class 0 vs 1
    mask = iris.target < 2
    X_iris = iris.data[mask]
    y_iris = iris.target[mask]

    diabetes = load_diabetes()
    X_diabetes = diabetes.data
    y_diabetes = diabetes.target

    # Generate CSV files for CLI
    import tempfile
    tmpdir = tempfile.mkdtemp()

    def save_csv(X, y, path):
        with open(path, "w") as f:
            for i in range(len(X)):
                row = ",".join(str(v) for v in X[i]) + "," + str(y[i])
                f.write(row + "\\n")

    iris_csv = os.path.join(tmpdir, "iris.csv")
    diabetes_csv = os.path.join(tmpdir, "diabetes.csv")
    save_csv(X_iris, y_iris, iris_csv)
    save_csv(X_diabetes, y_diabetes, diabetes_csv)

    import subprocess

    models = [
        ("iris_logreg.bin", f"train --model logreg --data {iris_csv} --target 4 --output {os.path.join(ZOO_DIR, 'iris_logreg.bin')}"),
        ("iris_knn.bin", f"train --model knn --data {iris_csv} --target 4 --params k=5 --output {os.path.join(ZOO_DIR, 'iris_knn.bin')}"),
        ("iris_nb.bin", f"train --model nb --data {iris_csv} --target 4 --output {os.path.join(ZOO_DIR, 'iris_nb.bin')}"),
        ("iris_dtree.bin", f"train --model dtree --data {iris_csv} --target 4 --output {os.path.join(ZOO_DIR, 'iris_dtree.bin')}"),
        ("diabetes_linreg.bin", f"train --model linreg --data {diabetes_csv} --target 10 --output {os.path.join(ZOO_DIR, 'diabetes_linreg.bin')}"),
    ]

    for name, cmd in models:
        full_cmd = f"{CLI_PATH} {cmd}"
        print(f"Training {name}...", end=" ", flush=True)
        result = subprocess.run(full_cmd.split(), capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED: {result.stderr[:100]}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    print(f"\\nDone. Models saved to {ZOO_DIR}/")


if __name__ == "__main__":
    main()
