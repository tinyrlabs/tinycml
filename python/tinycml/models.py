"""
tinycml.models — High-level sklearn-compatible Python API for tinycml

Usage:
    from tinycml import LinearRegression, LogisticRegression, KNNClassifier
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
"""

import numpy as np
from ._bindings import (
    LibTinyCML,
    np_to_matrix,
    matrix_to_np,
    require_2d,
)


class _Estimator:
    """Base class for all tinycml estimators."""

    _model_type = None  # override in subclass

    def __init__(self):
        self._handle = None
        self._lib = LibTinyCML()
        self._fitted = False

    def _ensure_model(self):
        if self._handle is None:
            self._handle = self._create_model()
            if self._handle is None:
                raise RuntimeError(f"Failed to create {self.__class__.__name__}")

    def _create_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        """Fit model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, 1)

        Returns
        -------
        self
        """
        X = require_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        self._ensure_model()
        n, d = X.shape
        X_mat = np_to_matrix(X)
        y_mat = np_to_matrix(y.reshape(-1, 1))

        result = self._lib.fit(self._handle, X_mat, y_mat)

        self._lib.matrix_free(X_mat)
        self._lib.matrix_free(y_mat)

        if result is None:
            raise RuntimeError(f"{self.__class__.__name__}.fit() failed")

        self._fitted = True
        self._n_features = d
        return self

    def predict(self, X):
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        X = require_2d(np.asarray(X, dtype=np.float64))
        X_mat = np_to_matrix(X)

        pred_mat = self._lib.predict(self._handle, X_mat)
        if pred_mat is None:
            raise RuntimeError(f"{self.__class__.__name__}.predict() failed")

        result = matrix_to_np(pred_mat).ravel()
        self._lib.matrix_free(pred_mat)
        self._lib.matrix_free(X_mat)
        return result

    def predict_proba(self, X):
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        X = require_2d(np.asarray(X, dtype=np.float64))
        X_mat = np_to_matrix(X)

        proba_mat = self._lib.predict_proba(self._handle, X_mat)
        if proba_mat is None:
            raise RuntimeError(f"{self.__class__.__name__}.predict_proba() failed")

        result = matrix_to_np(proba_mat)
        self._lib.matrix_free(proba_mat)
        self._lib.matrix_free(X_mat)
        return result

    def score(self, X, y):
        """Return the mean accuracy (classification) or R² (regression).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        X = require_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        X_mat = np_to_matrix(X)
        y_mat = np_to_matrix(y.reshape(-1, 1))

        s = self._lib.score(self._handle, X_mat, y_mat)

        self._lib.matrix_free(X_mat)
        self._lib.matrix_free(y_mat)
        return s

    def save(self, path):
        """Save model to binary file."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        rc = self._lib.save(self._handle, path.encode('utf-8'))
        if rc != 0:
            raise RuntimeError(f"Failed to save model to {path}")

    def load(self, path):
        """Load model from binary file."""
        handle = self._lib.load(path.encode('utf-8'))
        if handle is None:
            raise RuntimeError(f"Failed to load model from {path}")
        if self._handle is not None:
            self._lib.free(self._handle)
        self._handle = handle
        self._fitted = True
        return self

    def __del__(self):
        if self._handle is not None:
            try:
                self._lib.free(self._handle)
            except Exception:
                pass


# ─── Regression Models ────────────────────────────────────────────


class LinearRegression(_Estimator):
    """Ordinary least squares linear regression."""

    def __init__(self):
        super().__init__()

    def _create_model(self):
        return self._lib.linreg_create()


class Ridge(_Estimator):
    """L2-regularized linear regression."""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def _create_model(self):
        return self._lib.ridge_create(self.alpha)


class Lasso(_Estimator):
    """L1-regularized linear regression."""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _create_model(self):
        return self._lib.lasso_create(self.alpha, self.max_iter, self.tol)


class SGDRegressor(_Estimator):
    """Linear model trained with SGD (squared loss)."""

    def __init__(self, alpha=0.0001, learning_rate=0.01, max_iter=1000):
        super().__init__()
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _create_model(self):
        return self._lib.sgd_regressor_create(self.alpha, self.learning_rate, self.max_iter)


class DecisionTreeRegressor(_Estimator):
    """CART decision tree for regression."""

    def __init__(self, max_depth=10, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _create_model(self):
        return self._lib.dtree_reg_create(self.max_depth, self.min_samples_split)


class GradientBoostingRegressor(_Estimator):
    """Gradient Boosted Decision Trees for regression."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _create_model(self):
        return self._lib.gb_create(0, self.n_estimators, self.learning_rate, self.max_depth)


# ─── Classification Models ────────────────────────────────────────


class LogisticRegression(_Estimator):
    """Binary logistic regression classifier."""

    def __init__(self, learning_rate=0.01, max_iter=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _create_model(self):
        return self._lib.logreg_create(self.learning_rate, self.max_iter)


class KNNClassifier(_Estimator):
    """K-Nearest Neighbors classifier."""

    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def _create_model(self):
        return self._lib.knn_create(self.n_neighbors)


class SVMClassifier(_Estimator):
    """Support Vector Machine classifier (linear kernel)."""

    def __init__(self, C=1.0, max_iter=1000):
        super().__init__()
        self.C = C
        self.max_iter = max_iter

    def _create_model(self):
        return self._lib.svm_create(self.C, self.max_iter)


class GaussianNB(_Estimator):
    """Gaussian Naive Bayes classifier."""

    def __init__(self):
        super().__init__()

    def _create_model(self):
        return self._lib.nb_create()


class DecisionTreeClassifier(_Estimator):
    """CART decision tree for classification."""

    def __init__(self, max_depth=10, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _create_model(self):
        return self._lib.dtree_clf_create(self.max_depth, self.min_samples_split)


class RandomForestClassifier(_Estimator):
    """Random Forest classifier."""

    def __init__(self, n_estimators=100, max_depth=10):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _create_model(self):
        return self._lib.rf_create(self.n_estimators, self.max_depth)


class GradientBoostingClassifier(_Estimator):
    """Gradient Boosted Decision Trees for classification."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _create_model(self):
        return self._lib.gb_create(1, self.n_estimators, self.learning_rate, self.max_depth)


class SGDClassifier(_Estimator):
    """Linear classifier trained with SGD (log loss)."""

    def __init__(self, alpha=0.0001, learning_rate=0.01, max_iter=1000):
        super().__init__()
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _create_model(self):
        return self._lib.sgd_classifier_create(self.alpha, self.learning_rate, self.max_iter)
