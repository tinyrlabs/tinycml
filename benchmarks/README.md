# tinycml Benchmarks

This directory contains a benchmark suite that compares **tinycml** (C implementation) against **scikit-learn** (Python) on standard machine learning datasets.

## What is measured

- **Score** — Training accuracy (classification) or R² coefficient of determination (regression), computed on the training set for consistency.
- **Training time** — Wall-clock time to train the model, measured with `time.perf_counter()` (Python) and subprocess timing (tinycml CLI).

## Datasets

| Dataset | Task | Samples | Features |
|---------|------|---------|----------|
| iris | Classification | 150 | 4 |
| wine | Classification | 178 | 13 |
| breast_cancer | Classification | 569 | 30 |
| make_classification | Classification | 500 | 10 |
| california_housing | Regression | 500 | 8 |
| make_regression | Regression | 500 | 10 |

## Models

### Classification
logreg, knn, svm, nb, dtree, rf, gb, nn, softmax

### Regression
linreg, ridge, lasso, dtree, rf, gb, nn

## Prerequisites

1. **Build the tinycml CLI:**
   ```bash
   make cli
   ```
   This produces `build/bin/tinycml`.

2. **Install Python dependencies:**
   ```bash
   pip install numpy scikit-learn
   ```

## Running

From the project root:

```bash
make benchmark
```

Or directly:

```bash
python3 benchmarks/run_benchmarks.py
```

## Output

- **`benchmarks/results.md`** — Human-readable Markdown table with per-dataset comparisons
- **`benchmarks/results.csv`** — Machine-readable CSV for further analysis

Each row shows the model name, tinycml score, sklearn score, score difference, training times, and the speedup factor (sklearn time ÷ tinycml time).

## Notes

- Scores are computed on the **training set** because the tinycml CLI `train` command reports training metrics directly.
- For sklearn models, feature scaling (StandardScaler) is applied for models that benefit from it (SVM, NN, logistic regression, KNN).
- Temporary CSV files are created in `benchmarks/tmp/` and cleaned up after the run.
- The benchmark has a 300-second timeout per tinycml model.
