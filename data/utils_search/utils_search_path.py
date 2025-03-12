import os.path as op

import numpy as np

from emulator.utils_cache.utils_key import get_hash_str, get_hash_params
from utils.utils_path import SEARCH_CSV_PATH

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'
CHILDREN_FILENAME = 'children.txt'
PARENT_FILENAME = 'parent.txt'

def get_search_path(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool], non_default_params: dict) -> str:
    """Tree structure of the search_path is as follows: dataset_dir/emulator_folder"""
    return op.join(get_dataset_dir(X, y, validation_mask), get_emulator_folder(non_default_params))

def get_dataset_dir(X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool]):
    """Directory, containing subdirectories with search results, for a dataset and a validation size"""
    dataset_folder = get_hash_str(X, y, validation_mask)
    return op.join(SEARCH_CSV_PATH, dataset_folder)

def get_emulator_folder(non_default_params: dict) -> str:
    """Folder, whose name characterize the search using on default hyperparameters"""
    return get_hash_str(get_hash_params(non_default_params))