import math

import numpy as np
import pandas as pd


def compute_ind_validation(y_train: np.ndarray, validation_size: float, df: pd.DataFrame):
    return _compute_ind_validation(len(y_train), validation_size, load_nb_historical_values(df))


def _compute_ind_validation(length: int, validation_size: float, index_start_validation: int) -> np.ndarray[bool]:
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


def load_nb_historical_values(df: pd.DataFrame) -> int:
    """This index corresponds to the start of the rcp scenario, i.e. the start of the RCP scenario for training"""
    for index, (index_name, _) in enumerate(df.iterrows()):
        prefix = index_name[:3]
        if prefix == 'RCP':
            return index
        else:
            assert prefix == 'HIS'
    raise ValueError('No row of the dataframe starts with "RCP"')