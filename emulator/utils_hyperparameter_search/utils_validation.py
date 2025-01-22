import math

import numpy as np
import pandas as pd


def get_cv(ind_validation: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, validation_indices"""
    indices = np.arange(len(ind_validation))
    yield indices[~ind_validation], indices[ind_validation]


def compute_ind_validation(length: int, validation_size: float, index_start_validation: int) -> np.ndarray[bool]:
    """Compute an array of boolean such that ind_validation[i] = True if the index 'i' is in the validation set
    Parameters:
        length: int, length of the full time series
        validation_size: float, proportion (between 0 and 1) of data to include in the validation split
        index_start_validation: int, first index for the validation set
    Returns:
        ind_validation: np.ndarray[bool], ind_validation[i] = True if the index 'i' is in the validation set"""
    validation_length = math.ceil(length * validation_size)
    ind_validation = np.zeros(length).astype(bool)
    index_end_validation = validation_length + index_start_validation
    assert (0 <= index_start_validation) and (index_end_validation <= length)
    ind_validation[index_start_validation:index_end_validation] = True
    return ind_validation


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

