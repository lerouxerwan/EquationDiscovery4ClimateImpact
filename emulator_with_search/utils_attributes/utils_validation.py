import math

import numpy as np
import pandas as pd


def get_cv(ind_validation: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, validation_indices"""
    indices = np.arange(len(ind_validation))
    yield indices[~ind_validation], indices[ind_validation]


def get_X_and_y(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, ind_validation: np.ndarray, validation_set: bool):
    """Split X and y between train and validation.
    Returns X_train and y_train if validation_set=False other returns X_validation and y_validation"""
    if validation_set:
        if isinstance(X, np.ndarray):
            return X[ind_validation, :], y[ind_validation]
        else:
            return X.loc[ind_validation, :], y.loc[ind_validation]
    else:
        if isinstance(X, np.ndarray):
            return X[~ind_validation, :], y[~ind_validation]
        else:
            return X.loc[~ind_validation, :], y.loc[~ind_validation]

def apply_mask(X: np.ndarray | pd.DataFrame, mask: np.ndarray):
    assert mask is not None, mask
    assert len(mask) == X.shape[1], len(mask)
    if isinstance(X, np.ndarray):
        return X[:, mask]
    else:
        return X.loc[:, mask]

