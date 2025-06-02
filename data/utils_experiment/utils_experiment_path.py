import os.path as op
from typing import Optional

import numpy as np

from data.utils_experiment.utils_key import get_hash_str, get_hash_params
from utils.utils_path import EXPERIMENT_PATH

CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'
CHILDREN_FILENAME = 'children.txt'
PARENT_FILENAME = 'parent.txt'

def get_experiment_path(X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]], non_default_params: dict) -> str:
    """Tree structure of the experiment_path is as follows: dataset_dir/emulator_folder"""
    return op.join(get_dataset_dir(X, y, validation_mask), get_emulator_folder(non_default_params))

def get_dataset_dir(X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray[bool]]):
    """Directory, containing subdirectories with results, for a dataset and a validation size"""
    if validation_mask is None:
        dataset_folder = get_hash_str(X, y)
    else:
        dataset_folder = get_hash_str(X, y, validation_mask)
    return op.join(EXPERIMENT_PATH, dataset_folder)

def get_emulator_folder(non_default_params: dict) -> str:
    """Folder, whose name characterize the experiment using non default hyperparameters"""
    return get_hash_str(get_hash_params(non_default_params))