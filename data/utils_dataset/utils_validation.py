import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils_run import random_seed


def compute_default_validation_mask(y: np.ndarray) -> np.ndarray:
    """By default, create a random split with 30% and 70%"""
    indices = list(range(len(y)))
    _, indices_validation = train_test_split(np.array(indices), test_size=0.3, random_state=random_seed)
    indices_validation_set = set(indices_validation)
    return np.array([i in indices_validation_set for i in indices])

def compute_validation_mask(y_train: np.ndarray, validation_size: float, df: pd.DataFrame):
    return _compute_validation_mask(len(y_train), validation_size, load_nb_historical_values(df))

def _compute_validation_mask(length: int, validation_size: float, index_start_validation: int) -> np.ndarray[bool]:
    """Compute an array of boolean such that validation_mask[i] = True if the index 'i' is in the validation set
    Parameters:
        length: int, length of the full time series
        validation_size: float, proportion (between 0 and 1) of data to include in the validation split
        index_start_validation: int, first index for the validation set
    Returns:
        validation_mask: np.ndarray[bool], validation_mask[i] = True if the index 'i' is in the validation set"""
    validation_length = math.ceil(length * validation_size)
    validation_mask = np.zeros(length).astype(bool)
    index_end_validation = validation_length + index_start_validation
    assert (0 <= index_start_validation) and (index_end_validation <= length)
    validation_mask[index_start_validation:index_end_validation] = True
    return validation_mask


def load_nb_historical_values(df: pd.DataFrame) -> int:
    """This index corresponds to the start of the rcp scenario, i.e. the start of the RCP scenario for training"""
    for index, (index_name, _) in enumerate(df.iterrows()):
        prefix = index_name[:3]
        if prefix == 'RCP':
            return index
        else:
            assert prefix == 'HIS'
    raise ValueError('No row of the dataframe starts with "RCP"')