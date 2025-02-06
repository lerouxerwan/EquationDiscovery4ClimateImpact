import os.path as op
from typing import Optional, AnyStr, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from utils.utils_path import SEARCH_CSV_PATH

RANK_COLUMN_NAME = 'rank_test_MSE'
METRIC_COLUMN_NAME = 'mean_test_MSE'
CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'
DIGITS = 1

def get_folder_path(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float, feature_selection_name: str,
                    select_k_features: Optional[int], search_cv_type: type, n_iter: int, param_grid: dict, ) -> str:
    """Define a folder organization for a specific experiment"""
    experiment_path = get_experiment_path(X, y, validation_size, feature_selection_name, select_k_features)
    param_folder = get_param_folder(search_cv_type, n_iter, param_grid)
    return op.join(experiment_path, param_folder)


def get_param_folder(search_cv_type, n_iter, param_grid):
    param_folder = f'{search_cv_type.__name__}_{n_iter if search_cv_type is RandomizedSearchCV else ""}'
    param_folder += '_' + param_grid_signature(param_grid)
    return param_folder


def get_experiment_path(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float, feature_selection_name: str,
                    select_k_features: Optional[int]):
    dataset_search_path = get_dataset_search_path(X, y, validation_size)
    feature_selection_folder = f'{feature_selection_name}_{select_k_features}'
    return op.join(dataset_search_path, feature_selection_folder)

def get_dataset_search_path(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, validation_size: float):
    X_sum, y_sum = get_X_sum_and_y_sum(X, y)
    dataset_folder = f'{round(X_sum, DIGITS)}_{round(y_sum, DIGITS)}_{validation_size}'
    return op.join(SEARCH_CSV_PATH, dataset_folder)


def get_X_sum_and_y_sum(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> tuple[float, float]:
    X_sum, y_sum = (X.sum(), y.sum()) if isinstance(X, np.ndarray) else (X.values.sum(), y.values.sum())
    return float(X_sum), float(y_sum)


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

def get_non_default_params(estimator: BaseEstimator) -> dict[str, Any]:
    default_params = type(estimator)().get_params()
    return {k: v for k, v in estimator.get_params().items() if v != default_params[k]}