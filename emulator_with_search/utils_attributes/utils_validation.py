import math

import numpy as np
import pandas as pd


def get_cv(validation_mask: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, validation_indices"""
    indices = np.arange(len(validation_mask))
    yield indices[~validation_mask], indices[validation_mask]


def get_X_and_y(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray, validation_set: bool):
    """Split X and y between train and validation.
    Returns X_train and y_train if validation_set=False other returns X_validation and y_validation"""
    return (X[validation_mask, :], y[validation_mask]) if validation_set else (X[~validation_mask, :], y[~validation_mask])

def apply_mask(X: np.ndarray, mask: np.ndarray):
    assert mask is not None, mask
    assert len(mask) == X.shape[1], len(mask)
    if isinstance(X, np.ndarray):
        return X[:, mask]
    else:
        return X.loc[:, mask]

