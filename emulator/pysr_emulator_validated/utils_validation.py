import math

import numpy as np


def get_cv(ind_validation: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, test_indices"""
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
