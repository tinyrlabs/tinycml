"""Tests for tinycml Python bindings"""

import sys
import os
import tempfile

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Set library path before importing
os.environ.setdefault("TINYCML_LIB", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "build", "lib", "libtinycml.so.0.1.0"
))


def test_library_loads():
    """Test that the C library can be loaded."""
    from tinycml._bindings import LibTinyCML
    lib = LibTinyCML()
    assert lib._lib is not None
    print("  PASS  library loads")


def test_linear_regression():
    """Test LinearRegression fit/predict/score."""
    from tinycml.models import LinearRegression

    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.1  # y = 1.5*x0 - 2*x1 + 0.5*x2 + noise

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (100,)
    score = model.score(X, y)
    assert score > 0.95, f"R² = {score}, expected > 0.95"
    print(f"  PASS  LinearRegression R²={score:.4f}")


def test_logistic_regression():
    """Test LogisticRegression on linearly separable data."""
    from tinycml.models import LogisticRegression

    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    model = LogisticRegression(learning_rate=0.1, max_iter=500)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.85, f"Accuracy = {score}, expected > 0.85"
    print(f"  PASS  LogisticRegression accuracy={score:.4f}")


def test_knn_classifier():
    """Test KNNClassifier on simple data."""
    from tinycml.models import KNNClassifier

    np.random.seed(42)
    n = 60
    X = np.random.randn(n, 2)
    y = (X[:, 0] > 0).astype(float)

    model = KNNClassifier(n_neighbors=3)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.8, f"Accuracy = {score}, expected > 0.8"
    print(f"  PASS  KNNClassifier accuracy={score:.4f}")


def test_save_load():
    """Test model save/load."""
    from tinycml.models import LinearRegression

    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1]

    model1 = LinearRegression()
    model1.fit(X, y)
    score_before = model1.score(X, y)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        model1.save(path)
        model2 = LinearRegression()
        model2.load(path)
        score_after = model2.score(X, y)
        assert abs(score_before - score_after) < 0.01
        print(f"  PASS  save/load score_before={score_before:.4f} score_after={score_after:.4f}")
    finally:
        os.unlink(path)


def test_predict_proba():
    """Test predict_proba returns proper shape."""
    from tinycml.models import LogisticRegression

    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(float)

    model = LogisticRegression()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (50, 2), f"Shape = {proba.shape}"
    # Probabilities should sum to ~1
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=0.01), f"Sums = {sums[:5]}"
    print(f"  PASS  predict_proba shape={proba.shape}")


if __name__ == "__main__":
    print("=== tinycml Python Tests ===\n")
    tests = [
        test_library_loads,
        test_linear_regression,
        test_logistic_regression,
        test_knn_classifier,
        test_save_load,
        test_predict_proba,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed > 0:
        sys.exit(1)
