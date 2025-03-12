import os.path as op
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from emulator.utils_cache.utils_key import get_hash_str
from utils.utils_path import SEARCH_CSV_PATH

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'
CHILDREN_FILENAME = 'children.txt'
PARENT_FILENAME = 'parent.txt'
DIGITS = 1

"""
The tree structure of the search_path is as follows: dataset_dir/emulator_folder where:
    -dataset_dir is a path that characterizes a dataset and a validation setting
    -emulator_folder characterizes a hyperparameter search (search type, and non default hyperparameters)    
"""

def get_search_path(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, ind_validation: np.ndarray[bool],
                    search_style: str, n_iter: int, non_default_params: dict) -> str:
    feature_dir = get_dataset_dir(X, y, ind_validation)
    emulator_folder = get_emulator_folder(search_style, n_iter, non_default_params)
    return op.join(feature_dir, emulator_folder)

def get_dataset_dir(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, ind_validation: np.ndarray[bool]):
    """Directory, containing subdirectories with search results, for a dataset and a validation size"""
    dataset_folder = get_hash_str(X, y, ind_validation)
    return op.join(SEARCH_CSV_PATH, dataset_folder)

def get_emulator_folder(search_style: str, n_iter: int, non_default_params: dict) -> str:
    """Folder, whose name characterize the search (search type, number of iterations, non default hyperparameters)"""
    folder = f'{search_style}_{n_iter}'
    folder += '_' + search_signature_signature(non_default_params)
    return folder

def search_signature_signature(non_default_params: dict) -> str:
    keys_to_remove = ['search_style', 'n_iter', 'threshold_for_model_selection']
    for key_to_remove in keys_to_remove:
        if key_to_remove in non_default_params:
            non_default_params.pop(key_to_remove)
    # If param grid has been specified by the user, 'param_list_to_optimize' has its default value (None)
    param_grid_has_been_specified_by_user = 'param_list_to_optimize' not in non_default_params
    if not param_grid_has_been_specified_by_user:
        # In this case, we can remove 'param_grid' from the signature, because it can be deduced from the other infos
        non_default_params.pop('param_grid')
    # Create a unique signature, a string, containing all non default parameters
    short_name_to_hash_str = dict()
    short_names = set()
    for name, value in non_default_params.items():
        # Create unique short name
        name_without_backspace = name.replace('_', '')
        short_name = name_without_backspace[:3]
        if short_name in short_names:
            short_name += name_without_backspace[-3:]
            assert short_name not in short_names
        short_names.add(short_name)
        # Map short name to hash_str
        if isinstance(value, (int, float)):
            hash_str = short_name + str(value)
        elif isinstance(value, str):
            hash_str = short_name + value
        elif isinstance(value, list):
            hash_str = short_name + ''.join(['d' if s == '/' else str(s)[:1] for s in sorted(value)])
        elif isinstance(value, dict) and (name == 'param_grid'):
            hash_str = param_grid_signature(value)
        else:
            raise ValueError(f'value for {name} has type {type(value)}')
        short_name_to_hash_str[short_name] = hash_str
    short_names_sorted = [name for name in sorted(list(short_name_to_hash_str.keys()))]
    return '_'.join([short_name_to_hash_str[short_name] for short_name in short_names_sorted])

def param_grid_signature(param_grid: dict) -> str:
    efficient_param_grid = dict()
    short_names = set()
    for name, values in param_grid.items():
        if len(values) > 1:
            short_name = name[:3]
            if short_name in short_names:
                short_name += name[-3:]
            assert short_name not in short_names
            short_names.add(short_name)
            efficient_param_grid[short_name] = f'{round(min(values), DIGITS)}_{round(max(values), DIGITS)}'
    names_sorted = [name for name in sorted(list(efficient_param_grid.keys()))]
    return '_'.join([name + efficient_param_grid[name] for name in names_sorted])

def get_best_params(df_cv_results_ranked: pd.DataFrame) -> dict[str, Any]:
    return df_cv_results_ranked.iloc[0].loc['params']

def get_non_default_params(estimator: BaseEstimator) -> dict[str, Any]:
    """Return a dictionary that maps each the name of each non default parameter to its non default value"""
    default_params = type(estimator)().get_params()
    non_default_params = {}
    for param_name, param_value in estimator.get_params().items():
        default_value = default_params[param_name]
        if param_value != default_value:
            non_default_params[param_name] = param_value
    return non_default_params