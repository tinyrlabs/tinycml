#!/usr/bin/env python3
"""
tinycml vs sklearn Benchmark Suite
===================================
Compares tinycml (C CLI) against scikit-learn on standard datasets,
measuring accuracy/R² and training wall-clock time.

Usage:
    python3 benchmarks/run_benchmarks.py

Requirements:
    - Python 3.8+
    - numpy, scikit-learn
    - tinycml CLI built at build/bin/tinycml (run `make cli` first)

Output:
    benchmarks/results.md   — Markdown table
    benchmarks/results.csv  — CSV file
"""

import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TINYCML_CLI = PROJECT_ROOT / "build" / "bin" / "tinycml"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_MD = BENCHMARK_DIR / "results.md"
RESULTS_CSV = BENCHMARK_DIR / "results.csv"
TMP_DIR = BENCHMARK_DIR / "tmp"

# Models that can be evaluated via the tinycml CLI `train` command
# (name, is_classifier)
TINYCML_CLASSIFIERS = [
    "logreg",
    "knn",
    "svm",
    "nb",
    "dtree",
    "rf",
    "gb",
    "nn",
    "softmax",
]

TINYCML_REGRESSORS = [
    "linreg",
    "ridge",
    "lasso",
    # dtree, rf, gb also support regression but vary by dataset
]

# sklearn equivalents
SKLEARN_CLASSIFIERS = {
    "logreg": "sklearn.linear_model.LogisticRegression(max_iter=1000)",
    "knn": "sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)",
    "svm": "sklearn.svm.SVC(kernel='linear')",
    "nb": "sklearn.naive_bayes.GaussianNB()",
    "dtree": "sklearn.tree.DecisionTreeClassifier()",
    "rf": "sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=42)",
    "gb": "sklearn.ensemble.GradientBoostingClassifier(n_estimators=50, random_state=42)",
    "nn": "sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)",
    "softmax": "sklearn.linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000)",
}

SKLEARN_REGRESSORS = {
    "linreg": "sklearn.linear_model.LinearRegression()",
    "ridge": "sklearn.linear_model.Ridge()",
    "lasso": "sklearn.linear_model.Lasso()",
    "dtree": "sklearn.tree.DecisionTreeRegressor()",
    "rf": "sklearn.ensemble.RandomForestRegressor(n_estimators=10, random_state=42)",
    "gb": "sklearn.ensemble.GradientBoostingRegressor(n_estimators=50, random_state=42)",
    "nn": "sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    dataset: str
    model_name: str
    tinycml_score: Optional[float] = None
    sklearn_score: Optional[float] = None
    tinycml_time_ms: Optional[float] = None
    sklearn_time_ms: Optional[float] = None
    metric: str = ""  # "Accuracy" or "R²"
    tinycml_error: str = ""
    sklearn_error: str = ""

    @property
    def score_diff(self) -> Optional[float]:
        if self.tinycml_score is not None and self.sklearn_score is not None:
            return self.tinycml_score - self.sklearn_score
        return None

    @property
    def speedup(self) -> Optional[float]:
        if self.tinycml_time_ms and self.sklearn_time_ms:
            if self.tinycml_time_ms > 0:
                return self.sklearn_time_ms / self.tinycml_time_ms
        return None


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def load_datasets() -> List[Tuple[str, np.ndarray, np.ndarray, bool]]:
    """Return list of (name, X, y, is_classifier) tuples."""
    from sklearn import datasets as sk_datasets
    from sklearn.model_selection import train_test_split

    datasets = []

    # --- Classification datasets ---
    print("  Loading iris...")
    data = sk_datasets.load_iris()
    datasets.append(("iris", data.data, data.target, True))

    print("  Loading wine...")
    data = sk_datasets.load_wine()
    datasets.append(("wine", data.data, data.target, True))

    print("  Loading breast_cancer...")
    data = sk_datasets.load_breast_cancer()
    datasets.append(("breast_cancer", data.data, data.target, True))

    print("  Generating make_classification (500 samples, 10 features)...")
    X, y = sk_datasets.make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_classes=3, random_state=42,
    )
    datasets.append(("make_classification", X, y, True))

    # --- Regression datasets ---
    print("  Loading california_housing (500 subsample)...")
    data = sk_datasets.fetch_california_housing()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(data.data), size=500, replace=False)
    datasets.append(("california_housing", data.data[idx], data.target[idx], False))

    print("  Generating make_regression (500 samples, 10 features)...")
    X, y = sk_datasets.make_regression(
        n_samples=500, n_features=10, n_informative=5, noise=10.0, random_state=42,
    )
    datasets.append(("make_regression", X, y, False))

    return datasets


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Write a CSV with header that tinycml can load. Last column = target."""
    n_features = X.shape[1]
    header = ",".join([f"f{i}" for i in range(n_features)] + ["target"])
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(len(X)):
            row = ",".join(f"{v:.8f}" for v in X[i]) + f",{y[i]:.8f}"
            f.write(row + "\n")


# ---------------------------------------------------------------------------
# tinycml benchmarking
# ---------------------------------------------------------------------------

def run_tinycml(
    model_name: str,
    train_csv: Path,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Run tinycml train via CLI. Returns (score, time_ms, stderr_output).
    Parses the score (Accuracy or R²) and training time from stdout.
    """
    model_bin = TMP_DIR / f"model_{model_name}.bin"

    cmd = [
        str(TINYCML_CLI),
        "train",
        "--model", model_name,
        "--data", str(train_csv),
        "--output", str(model_bin),
    ]

    try:
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    except subprocess.TimeoutExpired:
        return None, None, "timeout (>300s)"
    except FileNotFoundError:
        return None, None, f"CLI not found at {TINYCML_CLI}"

    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        return None, None, f"exit code {proc.returncode}: {stderr.strip()[:200]}"

    # Parse score from output — match "Accuracy: 0.9667" or "R²: 0.8500"
    # (also handles "R2:" since some terminals mangle UTF-8)
    score = None
    m = re.search(r"(?:Accuracy|R²|R2)\s*:\s*([0-9]*\.?[0-9]+)", stdout)
    if m:
        score = float(m.group(1))

    # Also capture time reported by tinycml (but we use wall-clock instead)
    return score, elapsed_ms, ""


# ---------------------------------------------------------------------------
# sklearn benchmarking
# ---------------------------------------------------------------------------

def run_sklearn_classifier(model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    """Train sklearn classifier, return (accuracy, time_ms, error)."""
    import sklearn.linear_model
    import sklearn.neighbors
    import sklearn.svm
    import sklearn.naive_bayes
    import sklearn.tree
    import sklearn.ensemble
    import sklearn.neural_network

    constructors = {
        "logreg": lambda: sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42),
        "knn": lambda: sklearn.neighbors.KNeighborsClassifier(n_neighbors=5),
        "svm": lambda: sklearn.svm.SVC(kernel="linear"),
        "nb": lambda: sklearn.naive_bayes.GaussianNB(),
        "dtree": lambda: sklearn.tree.DecisionTreeClassifier(random_state=42),
        "rf": lambda: sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=42),
        "gb": lambda: sklearn.ensemble.GradientBoostingClassifier(n_estimators=50, random_state=42),
        "nn": lambda: sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        ),
        "softmax": lambda: sklearn.linear_model.LogisticRegression(
            multi_class="multinomial", max_iter=1000, random_state=42,
        ),
    }

    if model_name not in constructors:
        return None, None, f"unknown sklearn model: {model_name}"

    model = constructors[model_name]()
    try:
        from sklearn.utils.multiclass import type_of_target
        from sklearn.preprocessing import StandardScaler

        # Scale features for models that benefit from it
        if model_name in ("svm", "nn", "logreg", "softmax", "knn"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        start = time.perf_counter()
        model.fit(X_scaled, y)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        score = model.score(X_scaled, y)
        return score, elapsed_ms, ""
    except Exception as e:
        return None, None, str(e)[:200]


def run_sklearn_regressor(model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    """Train sklearn regressor, return (R², time_ms, error)."""
    import sklearn.linear_model
    import sklearn.tree
    import sklearn.ensemble
    import sklearn.neural_network

    constructors = {
        "linreg": lambda: sklearn.linear_model.LinearRegression(),
        "ridge": lambda: sklearn.linear_model.Ridge(random_state=42),
        "lasso": lambda: sklearn.linear_model.Lasso(random_state=42),
        "dtree": lambda: sklearn.tree.DecisionTreeRegressor(random_state=42),
        "rf": lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=10, random_state=42),
        "gb": lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=50, random_state=42),
        "nn": lambda: sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        ),
    }

    if model_name not in constructors:
        return None, None, f"unknown sklearn model: {model_name}"

    model = constructors[model_name]()
    try:
        from sklearn.preprocessing import StandardScaler

        if model_name in ("nn", "ridge", "lasso"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        start = time.perf_counter()
        model.fit(X_scaled, y)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        score = model.score(X_scaled, y)  # R²
        return score, elapsed_ms, ""
    except Exception as e:
        return None, None, str(e)[:200]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results_md(results: List[BenchmarkResult]) -> str:
    """Generate Markdown table from results."""
    lines = []
    lines.append("# tinycml vs sklearn Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("All scores are computed on the **training set** (consistent with tinycml CLI output).")
    lines.append("Classification → Accuracy. Regression → R².")
    lines.append("Time is wall-clock training time in **milliseconds**.")
    lines.append("")

    # Group by dataset
    by_dataset: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_dataset.setdefault(r.dataset, []).append(r)

    for dataset_name, dresults in by_dataset.items():
        metric = dresults[0].metric if dresults else "Score"
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append("| Model | tinycml Score | sklearn Score | Δ Score | tinycml (ms) | sklearn (ms) | Speedup |")
        lines.append("|-------|--------------|--------------|---------|-------------|-------------|---------|")

        for r in dresults:
            tc_score = f"{r.tinycml_score:.4f}" if r.tinycml_score is not None else "—"
            sk_score = f"{r.sklearn_score:.4f}" if r.sklearn_score is not None else "—"
            delta = f"{r.score_diff:+.4f}" if r.score_diff is not None else "—"
            tc_time = f"{r.tinycml_time_ms:.2f}" if r.tinycml_time_ms is not None else "—"
            sk_time = f"{r.sklearn_time_ms:.2f}" if r.sklearn_time_ms is not None else "—"
            speedup = f"{r.speedup:.1f}x" if r.speedup is not None else "—"

            lines.append(f"| {r.model_name} | {tc_score} | {sk_score} | {delta} | {tc_time} | {sk_time} | {speedup} |")

        lines.append("")

    # Summary statistics
    all_tc_times = [r.tinycml_time_ms for r in results if r.tinycml_time_ms is not None]
    all_sk_times = [r.sklearn_time_ms for r in results if r.sklearn_time_ms is not None]
    all_scores = [r.score_diff for r in results if r.score_diff is not None]

    if all_tc_times and all_sk_times:
        avg_tc = np.mean(all_tc_times)
        avg_sk = np.mean(all_sk_times)
        avg_delta = np.mean(all_scores) if all_scores else 0.0
        median_speedup = np.median([r.speedup for r in results if r.speedup is not None])

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Average tinycml time:** {avg_tc:.2f} ms")
        lines.append(f"- **Average sklearn time:** {avg_sk:.2f} ms")
        lines.append(f"- **Median speedup (sklearn/tinycml):** {median_speedup:.1f}x")
        lines.append(f"- **Average score difference (tinycml − sklearn):** {avg_delta:+.4f}")
        lines.append("")

    return "\n".join(lines)


def write_results_csv(results: List[BenchmarkResult], path: Path) -> None:
    """Write results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "model", "metric",
            "tinycml_score", "sklearn_score", "score_diff",
            "tinycml_time_ms", "sklearn_time_ms", "speedup",
            "tinycml_error", "sklearn_error",
        ])
        for r in results:
            writer.writerow([
                r.dataset,
                r.model_name,
                r.metric,
                f"{r.tinycml_score:.6f}" if r.tinycml_score is not None else "",
                f"{r.sklearn_score:.6f}" if r.sklearn_score is not None else "",
                f"{r.score_diff:.6f}" if r.score_diff is not None else "",
                f"{r.tinycml_time_ms:.3f}" if r.tinycml_time_ms is not None else "",
                f"{r.sklearn_time_ms:.3f}" if r.sklearn_time_ms is not None else "",
                f"{r.speedup:.3f}" if r.speedup is not None else "",
                r.tinycml_error,
                r.sklearn_error,
            ])


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  tinycml vs sklearn Benchmark Suite")
    print("=" * 70)

    # Check CLI exists
    if not TINYCML_CLI.exists():
        print(f"\nERROR: tinycml CLI not found at {TINYCML_CLI}")
        print("Run `make cli` first to build the CLI tool.")
        sys.exit(1)

    # Create directories
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n[1/4] Loading datasets...")
    datasets = load_datasets()

    all_results: List[BenchmarkResult] = []

    # For each dataset, run all applicable models
    print("\n[2/4] Running benchmarks...\n")

    total_benchmarks = 0
    for dataset_name, X, y, is_classifier in datasets:
        n_samples, n_features = X.shape
        task = "classification" if is_classifier else "regression"
        print(f"--- Dataset: {dataset_name} ({n_samples} samples, {n_features} features, {task}) ---")

        # Write CSV for tinycml
        train_csv = TMP_DIR / f"{dataset_name}_train.csv"
        write_csv(train_csv, X, y)
        print(f"  Wrote {train_csv}")

        # Select models
        if is_classifier:
            models = TINYCML_CLASSIFIERS
            metric = "Accuracy"
        else:
            models = TINYCML_REGRESSORS
            metric = "R²"

        for model_name in models:
            total_benchmarks += 1
            print(f"  [{model_name:>10s}] ", end="", flush=True)
            result = BenchmarkResult(
                dataset=dataset_name,
                model_name=model_name,
                metric=metric,
            )

            # --- tinycml ---
            tc_score, tc_time, tc_err = run_tinycml(model_name, train_csv)
            result.tinycml_score = tc_score
            result.tinycml_time_ms = tc_time
            result.tinycml_error = tc_err

            if tc_err:
                print(f"tinycml=ERR({tc_err[:40]}) ", end="", flush=True)
            else:
                tc_str = f"{tc_score:.4f}" if tc_score is not None else "N/A"
                print(f"tinycml={tc_str}/{tc_time:.1f}ms ", end="", flush=True)

            # --- sklearn ---
            if is_classifier:
                sk_score, sk_time, sk_err = run_sklearn_classifier(model_name, X, y)
            else:
                sk_score, sk_time, sk_err = run_sklearn_regressor(model_name, X, y)
            result.sklearn_score = sk_score
            result.sklearn_time_ms = sk_time
            result.sklearn_error = sk_err

            if sk_err:
                print(f"sklearn=ERR({sk_err[:40]})")
            else:
                sk_str = f"{sk_score:.4f}" if sk_score is not None else "N/A"
                print(f"sklearn={sk_str}/{sk_time:.1f}ms")

            all_results.append(result)

    # Also run some regression models on regression datasets (dtree, rf, gb)
    for dataset_name, X, y, is_classifier in datasets:
        if is_classifier:
            continue
        for model_name in ("dtree", "rf", "gb"):
            total_benchmarks += 1
            print(f"  [{model_name:>10s}] ", end="", flush=True)
            result = BenchmarkResult(
                dataset=dataset_name,
                model_name=model_name,
                metric="R²",
            )
            train_csv = TMP_DIR / f"{dataset_name}_train.csv"

            # tinycml — these ensemble models may handle regression
            tc_score, tc_time, tc_err = run_tinycml(model_name, train_csv)
            result.tinycml_score = tc_score
            result.tinycml_time_ms = tc_time
            result.tinycml_error = tc_err

            if tc_err:
                print(f"tinycml=ERR({tc_err[:40]}) ", end="", flush=True)
            else:
                tc_str = f"{tc_score:.4f}" if tc_score is not None else "N/A"
                print(f"tinycml={tc_str}/{tc_time:.1f}ms ", end="", flush=True)

            # sklearn
            sk_score, sk_time, sk_err = run_sklearn_regressor(model_name, X, y)
            result.sklearn_score = sk_score
            result.sklearn_time_ms = sk_time
            result.sklearn_error = sk_err

            if sk_err:
                print(f"sklearn=ERR({sk_err[:40]})")
            else:
                sk_str = f"{sk_score:.4f}" if sk_score is not None else "N/A"
                print(f"sklearn={sk_str}/{sk_time:.1f}ms")

            all_results.append(result)

    print(f"\n  Total benchmarks: {total_benchmarks}")

    # Write output
    print("\n[3/4] Writing results...")

    md_content = format_results_md(all_results)
    RESULTS_MD.write_text(md_content)
    print(f"  Markdown → {RESULTS_MD}")

    write_results_csv(all_results, RESULTS_CSV)
    print(f"  CSV       → {RESULTS_CSV}")

    # Cleanup temp files
    print("\n[4/4] Cleaning up...")
    for f in TMP_DIR.iterdir():
        f.unlink()
    TMP_DIR.rmdir()

    # Print summary
    print("\n" + "=" * 70)
    print("  Benchmark complete!")
    print("=" * 70)
    print(f"\nResults written to:")
    print(f"  {RESULTS_MD}")
    print(f"  {RESULTS_CSV}")

    # Quick summary
    n_tinycml_ok = sum(1 for r in all_results if r.tinycml_score is not None)
    n_sklearn_ok = sum(1 for r in all_results if r.sklearn_score is not None)
    print(f"\n  tinycml: {n_tinycml_ok}/{len(all_results)} successful")
    print(f"  sklearn: {n_sklearn_ok}/{len(all_results)} successful")


if __name__ == "__main__":
    main()
