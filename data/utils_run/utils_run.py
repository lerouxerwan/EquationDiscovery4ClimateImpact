import os.path as op
from typing import Optional

import numpy as np

from data.utils_run.utils_key import get_hash_str, get_hash_params
from utils.utils_path import RUN_PATH

CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'

def get_output_directory(X: np.ndarray, y: np.ndarray, validation_mask: Optional[np.ndarray]):
    """Directory, containing subdirectories with results, for a dataset and a validation size"""
    if validation_mask is None:
        dataset_folder = get_hash_str(X, y)
    else:
        dataset_folder = get_hash_str(X, y, validation_mask)
    return op.join(RUN_PATH, dataset_folder)

def get_run_id(non_default_params: dict) -> str:
    """Folder, whose name characterize the run using non default hyperparameters"""
    return get_hash_str(get_hash_params(non_default_params))