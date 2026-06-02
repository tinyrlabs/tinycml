"""
Low-level ctypes bindings for the tinycml shared library.

This module loads ``libtinycml.so`` and wraps every C function that the
high-level Python API needs.  It is *not* meant to be used directly by
end-users — they should use the classes in ``tinycml.models`` instead.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None


def _search_paths() -> list[str]:
    """Return candidate paths for libtinycml.so in priority order."""
    # 1. Explicit env var
    env = os.environ.get("TINYCML_LIB")
    if env:
        return [env]

    # 2. Relative to this file → project build/lib/
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # python/tinycml/_bindings.py → project root

    candidates = [
        project_root / "build" / "lib" / "libtinycml.so",
        project_root / "build" / "lib" / "libtinycml.so.0.1.0",
        Path("/usr/local/lib/libtinycml.so"),
        Path("/usr/local/lib/libtinycml.so.0.1.0"),
        Path("libtinycml.so"),
    ]

    # Also try a few system paths
    for p in ("/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib"):
        candidates.append(Path(p) / "libtinycml.so")

    return [str(c) for c in candidates]


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    errors: list[str] = []
    for path in _search_paths():
        try:
            _LIB = ctypes.CDLL(path)
            return _LIB
        except OSError as exc:
            errors.append(f"  {path}: {exc}")

    raise OSError(
        "Could not load libtinycml.so.\n"
        "Set the TINYCML_LIB environment variable to the full path of the "
        "shared library, or build the project with `make shared`.\n"
        "Paths tried:\n" + "\n".join(errors)
    )


def get_lib() -> ctypes.CDLL:
    """Return the loaded library, loading it on first call."""
    return _load_library()


# ---------------------------------------------------------------------------
# ctypes structures matching the C headers
# ---------------------------------------------------------------------------

class CMatrix(ctypes.Structure):
    """Maps to the C ``Matrix`` struct (matrix.h)."""
    _fields_ = [
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.c_double)),
    ]


class CEstimator(ctypes.Structure):
    """Maps to the C ``Estimator`` struct (estimator.h).

    We only need the first few fields to detect fitted status and to call
    vtable functions through the C function wrappers (not through the
    embedded function pointers, which are complex to bind via ctypes).
    """
    _fields_ = [
        ("type", ctypes.c_int),    # ModelType enum
        ("task", ctypes.c_int),    # TaskType enum
        ("is_fitted", ctypes.c_int),
    ]


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _numpy_to_matrix(arr) -> ctypes.POINTER(CMatrix):
    """Convert a 2-D numpy array to a C ``Matrix*`` allocated via the library.

    The returned pointer owns the C memory and must be freed with
    ``matrix_free`` when no longer needed.
    """
    import numpy as np

    lib = get_lib()
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    rows, cols = arr.shape
    mat_ptr = lib.matrix_alloc(ctypes.c_size_t(rows), ctypes.c_size_t(cols))
    if not mat_ptr:
        raise MemoryError("matrix_alloc returned NULL")

    # Copy data row-by-row (row-major, same layout as numpy)
    for i in range(rows):
        for j in range(cols):
            lib.matrix_set(
                mat_ptr,
                ctypes.c_size_t(i),
                ctypes.c_size_t(j),
                ctypes.c_double(arr[i, j]),
            )
    return mat_ptr


def _matrix_to_numpy(mat_ptr) -> "numpy.ndarray":
    """Copy a C ``Matrix*`` into a numpy array and free the C matrix."""
    import numpy as np

    lib = get_lib()
    mat = mat_ptr.contents
    rows, cols = mat.rows, mat.cols
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = lib.matrix_get(mat_ptr, ctypes.c_size_t(i), ctypes.c_size_t(j))
    lib.matrix_free(mat_ptr)
    if out.shape[1] == 1:
        return out.ravel()
    return out


def _free_matrix(mat_ptr):
    """Free a C Matrix* (tolerates None)."""
    if mat_ptr:
        get_lib().matrix_free(mat_ptr)


# ---------------------------------------------------------------------------
# Generic Estimator vtable call helpers
# ---------------------------------------------------------------------------

def _estimator_fit(estimator_ptr, X_mat, y_mat):
    """Call the Estimator's fit vtable entry."""
    lib = get_lib()
    # We use the C-level fit functions directly (not the vtable pointer)
    # to keep the binding simple and portable.
    # The individual model wrappers below call the model-specific fit.
    # This generic version exists as a fallback.
    result = lib.matrix_alloc(ctypes.c_size_t(0), ctypes.c_size_t(0))  # placeholder
    return estimator_ptr


def _estimator_predict(estimator_ptr, X_mat):
    """Call the Estimator's predict vtable entry.

    This invokes the predict function pointer stored in the Estimator base.
    """
    lib = get_lib()

    # Read the predict function pointer from offset in the struct.
    # Estimator layout: type(4) + task(4) + is_fitted(4) + padding(4) +
    #   fit(8) + predict(8) + ...
    # On 64-bit, after int fields (12 bytes padded to 16), then function ptrs.
    # This is fragile, so we prefer model-specific wrappers below.
    raise RuntimeError("Use model-specific predict wrappers instead of the generic vtable.")


# ---------------------------------------------------------------------------
# Wrapped C functions
# ---------------------------------------------------------------------------

def _setup_function_signatures(lib: ctypes.CDLL):
    """Set argtypes / restype for all wrapped C functions."""
    c_size_t = ctypes.c_size_t
    c_double = ctypes.c_double
    c_int = ctypes.c_int
    c_char_p = ctypes.c_char_p
    PMatrix = ctypes.POINTER(CMatrix)

    # --- matrix.h ---
    lib.matrix_alloc.argtypes = [c_size_t, c_size_t]
    lib.matrix_alloc.restype = PMatrix

    lib.matrix_free.argtypes = [PMatrix]
    lib.matrix_free.restype = None

    lib.matrix_copy.argtypes = [PMatrix]
    lib.matrix_copy.restype = PMatrix

    lib.matrix_get.argtypes = [PMatrix, c_size_t, c_size_t]
    lib.matrix_get.restype = c_double

    lib.matrix_set.argtypes = [PMatrix, c_size_t, c_size_t, c_double]
    lib.matrix_set.restype = None

    # --- linear_regression.h ---
    lib.linear_regression_create.argtypes = [c_int]
    lib.linear_regression_create.restype = ctypes.c_void_p

    lib.linear_regression_create_full.argtypes = [
        c_int, c_double, c_int, c_double, c_int, c_int
    ]
    lib.linear_regression_create_full.restype = ctypes.c_void_p

    lib.linear_regression_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.linear_regression_fit.restype = ctypes.c_void_p

    lib.linear_regression_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.linear_regression_predict.restype = PMatrix

    lib.linear_regression_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.linear_regression_score.restype = c_double

    lib.linear_regression_free.argtypes = [ctypes.c_void_p]
    lib.linear_regression_free.restype = None

    lib.linear_regression_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.linear_regression_save.restype = c_int

    lib.linear_regression_load.argtypes = [c_char_p]
    lib.linear_regression_load.restype = ctypes.c_void_p

    # --- logistic_regression.h ---
    lib.logreg_model_create.argtypes = []
    lib.logreg_model_create.restype = ctypes.c_void_p

    lib.logreg_model_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.logreg_model_save.restype = c_int

    lib.logreg_model_load.argtypes = [c_char_p]
    lib.logreg_model_load.restype = ctypes.c_void_p

    # --- ridge.h ---
    lib.ridge_model_create.argtypes = []
    lib.ridge_model_create.restype = ctypes.c_void_p

    lib.ridge_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.ridge_fit.restype = ctypes.c_void_p

    lib.ridge_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.ridge_predict.restype = PMatrix

    lib.ridge_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.ridge_score.restype = c_double

    lib.ridge_free.argtypes = [ctypes.c_void_p]
    lib.ridge_free.restype = None

    # --- lasso.h ---
    lib.lasso_model_create.argtypes = []
    lib.lasso_model_create.restype = ctypes.c_void_p

    lib.lasso_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.lasso_fit.restype = ctypes.c_void_p

    lib.lasso_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.lasso_predict.restype = PMatrix

    lib.lasso_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.lasso_score.restype = c_double

    lib.lasso_free.argtypes = [ctypes.c_void_p]
    lib.lasso_free.restype = None

    # --- knn.h ---
    lib.knn_fit.argtypes = [PMatrix, PMatrix, c_int]
    lib.knn_fit.restype = ctypes.c_void_p

    lib.knn_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.knn_predict.restype = PMatrix

    lib.knn_predict_proba.argtypes = [ctypes.c_void_p, PMatrix, c_int]
    lib.knn_predict_proba.restype = PMatrix

    lib.knn_free.argtypes = [ctypes.c_void_p]
    lib.knn_free.restype = None

    # --- svm.h ---
    lib.linear_svc_create.argtypes = []
    lib.linear_svc_create.restype = ctypes.c_void_p

    lib.linear_svc_free_impl.argtypes = [ctypes.c_void_p]
    lib.linear_svc_free_impl.restype = None

    lib.linear_svc_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.linear_svc_save.restype = c_int

    lib.linear_svc_load.argtypes = [c_char_p]
    lib.linear_svc_load.restype = ctypes.c_void_p

    lib.svm_classifier_create.argtypes = [c_int]
    lib.svm_classifier_create.restype = ctypes.c_void_p

    lib.svm_classifier_free_impl.argtypes = [ctypes.c_void_p]
    lib.svm_classifier_free_impl.restype = None

    lib.svm_classifier_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.svm_classifier_save.restype = c_int

    lib.svm_classifier_load.argtypes = [c_char_p]
    lib.svm_classifier_load.restype = ctypes.c_void_p

    # --- naive_bayes.h ---
    lib.gaussian_nb_create.argtypes = []
    lib.gaussian_nb_create.restype = ctypes.c_void_p

    lib.gaussian_nb_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.gaussian_nb_save.restype = c_int

    lib.gaussian_nb_load.argtypes = [c_char_p]
    lib.gaussian_nb_load.restype = ctypes.c_void_p

    # --- decision_tree.h ---
    lib.decision_tree_classifier_create.argtypes = []
    lib.decision_tree_classifier_create.restype = ctypes.c_void_p

    lib.decision_tree_classifier_create_full.argtypes = [
        c_int, c_int, c_int, c_int, c_double
    ]
    lib.decision_tree_classifier_create_full.restype = ctypes.c_void_p

    lib.decision_tree_classifier_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.decision_tree_classifier_fit.restype = ctypes.c_void_p

    lib.decision_tree_classifier_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.decision_tree_classifier_predict.restype = PMatrix

    lib.decision_tree_classifier_predict_proba.argtypes = [ctypes.c_void_p, PMatrix]
    lib.decision_tree_classifier_predict_proba.restype = PMatrix

    lib.decision_tree_classifier_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.decision_tree_classifier_score.restype = c_double

    lib.decision_tree_classifier_free.argtypes = [ctypes.c_void_p]
    lib.decision_tree_classifier_free.restype = None

    lib.decision_tree_regressor_create.argtypes = []
    lib.decision_tree_regressor_create.restype = ctypes.c_void_p

    lib.decision_tree_regressor_create_full.argtypes = [
        c_int, c_int, c_int, c_int, c_double
    ]
    lib.decision_tree_regressor_create_full.restype = ctypes.c_void_p

    lib.decision_tree_regressor_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.decision_tree_regressor_fit.restype = ctypes.c_void_p

    lib.decision_tree_regressor_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.decision_tree_regressor_predict.restype = PMatrix

    lib.decision_tree_regressor_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.decision_tree_regressor_score.restype = c_double

    lib.decision_tree_regressor_free.argtypes = [ctypes.c_void_p]
    lib.decision_tree_regressor_free.restype = None

    # --- ensemble.h ---
    lib.random_forest_classifier_create.argtypes = [c_int]
    lib.random_forest_classifier_create.restype = ctypes.c_void_p

    lib.random_forest_classifier_create_full.argtypes = [
        c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_uint
    ]
    lib.random_forest_classifier_create_full.restype = ctypes.c_void_p

    lib.random_forest_classifier_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.random_forest_classifier_fit.restype = ctypes.c_void_p

    lib.random_forest_classifier_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.random_forest_classifier_predict.restype = PMatrix

    lib.random_forest_classifier_predict_proba.argtypes = [ctypes.c_void_p, PMatrix]
    lib.random_forest_classifier_predict_proba.restype = PMatrix

    lib.random_forest_classifier_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.random_forest_classifier_score.restype = c_double

    lib.random_forest_classifier_free.argtypes = [ctypes.c_void_p]
    lib.random_forest_classifier_free.restype = None

    # --- gradient_boosting.h ---
    lib.gradient_boosting_create.argtypes = [c_int, c_double, c_int, c_int, c_double]
    lib.gradient_boosting_create.restype = ctypes.c_void_p

    lib.gradient_boosting_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.gradient_boosting_fit.restype = ctypes.c_void_p

    lib.gradient_boosting_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.gradient_boosting_predict.restype = PMatrix

    lib.gradient_boosting_predict_proba.argtypes = [ctypes.c_void_p, PMatrix]
    lib.gradient_boosting_predict_proba.restype = PMatrix

    lib.gradient_boosting_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.gradient_boosting_score.restype = c_double

    lib.gradient_boosting_free.argtypes = [ctypes.c_void_p]
    lib.gradient_boosting_free.restype = None

    lib.gradient_boosting_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.gradient_boosting_save.restype = c_int

    lib.gradient_boosting_load.argtypes = [c_char_p]
    lib.gradient_boosting_load.restype = ctypes.c_void_p

    # --- sgd.h ---
    lib.sgd_classifier_create.argtypes = [c_double, c_double, c_int]
    lib.sgd_classifier_create.restype = ctypes.c_void_p

    lib.sgd_regressor_create.argtypes = [c_double, c_double, c_int]
    lib.sgd_regressor_create.restype = ctypes.c_void_p

    lib.sgd_fit.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.sgd_fit.restype = ctypes.c_void_p

    lib.sgd_predict.argtypes = [ctypes.c_void_p, PMatrix]
    lib.sgd_predict.restype = PMatrix

    lib.sgd_predict_proba.argtypes = [ctypes.c_void_p, PMatrix]
    lib.sgd_predict_proba.restype = PMatrix

    lib.sgd_score.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.sgd_score.restype = c_double

    lib.sgd_free.argtypes = [ctypes.c_void_p]
    lib.sgd_free.restype = None

    lib.sgd_save.argtypes = [ctypes.c_void_p, c_char_p]
    lib.sgd_save.restype = c_int

    lib.sgd_load.argtypes = [c_char_p]
    lib.sgd_load.restype = ctypes.c_void_p

    # --- Generic estimator score helpers ---
    lib.classification_score_accuracy.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.classification_score_accuracy.restype = c_double

    lib.regression_score_r2.argtypes = [ctypes.c_void_p, PMatrix, PMatrix]
    lib.regression_score_r2.restype = c_double


def _ensure_signatures():
    """Lazily set up function signatures once after the library is loaded."""
    lib = get_lib()
    # Use a sentinel attribute on the lib object to avoid repeated setup.
    if not getattr(lib, "_tinycml_signatures_set", False):
        _setup_function_signatures(lib)
        lib._tinycml_signatures_set = True


# ---------------------------------------------------------------------------
# Public helpers used by models.py
# ---------------------------------------------------------------------------

def to_matrix(arr):
    """numpy array → C Matrix pointer (caller owns the memory)."""
    _ensure_signatures()
    return _numpy_to_matrix(arr)


def from_matrix(mat_ptr):
    """C Matrix pointer → numpy array (frees the C matrix)."""
    _ensure_signatures()
    return _matrix_to_numpy(mat_ptr)


def free_matrix(mat_ptr):
    """Free a C Matrix pointer."""
    _ensure_signatures()
    _free_matrix(mat_ptr)


def lib():
    """Return the loaded and configured ctypes library handle."""
    _ensure_signatures()
    return get_lib()
