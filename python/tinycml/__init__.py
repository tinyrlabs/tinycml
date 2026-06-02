"""
tinycml — Python bindings for the tinycml C ML library.

Provides sklearn-compatible estimators backed by the tinycml shared library
via ctypes (zero external dependencies beyond numpy for array I/O).
"""

__version__ = "0.1.0"

from tinycml.models import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    KNNClassifier,
    SVMClassifier,
    GaussianNB,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    SGDClassifier,
    SGDRegressor,
)

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "KNNClassifier",
    "SVMClassifier",
    "GaussianNB",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "SGDClassifier",
    "SGDRegressor",
]
