import os.path as op
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from emulator.utils_cache.utils_key import get_X_sum_and_y_sum
from utils.utils_path import SEARCH_CSV_PATH

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'
DIGITS = 1

"""
The tree structure of the search folders is as follows: dataset_dir/feature_folder/search_folder where:
    -dataset_dir is a path that characterizes a dataset and a validation setting
    -feature_folder is a folder that characterizes a feature_selection_method and some number of features
    -search_folder is a folder that characterizes a hyperparameter search (search type, and non default hyperparameters)    
"""

def get_dataset_dir(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float):
    """Directory, containing subdirectories with search results, for a dataset and a validation size"""
    X_sum, y_sum = get_X_sum_and_y_sum(X, y)
    dataset_folder = f'{round(X_sum, DIGITS)}_{round(y_sum, DIGITS)}_{validation_size}'
    return op.join(SEARCH_CSV_PATH, dataset_folder)

def get_feature_dir(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float,
                    feature_selection_name: str, select_k_features: Optional[int]) -> str:
    """Directory, containing subdirectories with search results, for a feature selection and number of features"""
    dataset_dir = get_dataset_dir(X, y, validation_size)
    feature_folder = f'{feature_selection_name}_{select_k_features}'
    return op.join(dataset_dir, feature_folder)

def get_search_folder(search_cv_type: type, n_iter: int, non_default_params: dict) -> str:
    """Folder, whose name characterize the search (search type, number of iterations, non default hyperparameters)"""
    # Potentially remove feature selection and number of features from non default params
    # because these two settings are already contained in the search_dir
    for param_name in ['feature_selection_name', 'select_k_features']:
        if param_name in non_default_params:
            non_default_params.pop(param_name)
    search_folder = f'{search_cv_type.__name__}_{n_iter if search_cv_type is RandomizedSearchCV else ""}'
    search_folder += '_' + search_signature_signature(non_default_params)
    return search_folder

def search_signature_signature(non_default_params: dict) -> str:
    # If param grid has been specified by the user, 'param_list_to_optimize_around_default' has its default value (None)
    param_grid_has_been_specified_by_user = 'param_list_to_optimize_around_default' not in non_default_params
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
    return {k: v for k, v in estimator.get_params().items() if v != default_params[k]}