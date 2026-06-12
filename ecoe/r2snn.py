"""Reusable R2SNN surrogate helpers.

This module wraps the implementation used by the experiment scripts so external
users can import the proposed algorithm without depending on the runners.
"""

from src.r2snn import R2SNN, train_r2snn


def fit_surrogate(predictor, X_train, **kwargs):
    """Fit an R2SNN surrogate for a classifier-like object."""
    return train_r2snn(predictor, X_train, **kwargs)
