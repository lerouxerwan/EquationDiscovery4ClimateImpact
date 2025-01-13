import math

import numpy as np


def get_cv(ind_validation: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, test_indices"""
    indices = np.arange(len(ind_validation))
    yield indices[~ind_validation], indices[ind_validation]


def compute_ind_validation(length: int, validation_size: float, start_index_of_validation: int) -> np.ndarray[bool]:
    """Compute an array of boolean such that ind_validation[i] = True if the index 'i' is in the validation set
    Parameters:
        length: int, length of the full time series
        validation_size: float, proportion (between 0 and 1) of data to include in the validation split
        start_index_of_validation: int, first index for the validation set
    Returns:
        ind_validation: np.ndarray[bool], ind_validation[i] = True if the index 'i' is in the validation set"""
    validation_length = math.ceil(length * validation_size)
    ind_validation = np.zeros(length).astype(bool)
    end_index_of_validation = validation_length + start_index_of_validation
    assert (0 <= start_index_of_validation) and (end_index_of_validation <= length)
    ind_validation[start_index_of_validation:end_index_of_validation] = True
    return ind_validation
