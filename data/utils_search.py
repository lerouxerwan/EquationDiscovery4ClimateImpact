import os.path as op
from typing import Optional, AnyStr, Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from utils.utils_path import SEARCH_CSV_PATH

DIGITS = 1


def get_folder_path(X_sum: float, y_sum: float, validation_size: float, search_cv_type: type, n_iter: int,
                        param_grid: dict, feature_selection_name: str, select_k_features: Optional[int]) -> str:
    """Define a folder organization for a specific experiment"""
    dataset_folder = f'{round(X_sum, DIGITS)}_{round(y_sum, DIGITS)}_{validation_size}'
    feature_selection_folder = f'{feature_selection_name}_{select_k_features}'
    param_folder = f'{search_cv_type.__name__}_{n_iter if search_cv_type is RandomizedSearchCV else ""}'
    param_folder += '_' + param_grid_signature(param_grid)
    return op.join(SEARCH_CSV_PATH, dataset_folder, feature_selection_folder, param_folder)


def param_grid_signature(param_grid: dict) -> str:
    efficient_param_grid = {}
    for name, values in param_grid.items():
        if len(values) > 1:
            efficient_param_grid[name[:3]] = f'{round(min(values), DIGITS)}_{round(max(values), DIGITS)}'
    names_sorted = [name for name in sorted(list(efficient_param_grid.keys()))]
    return '_'.join([name + efficient_param_grid[name] for name in names_sorted])

def get_non_default_params(estimator: BaseEstimator) -> dict[str, Any]:
    default_params = type(estimator)().get_params()
    return {k: v for k, v in estimator.get_params().items() if v != default_params[k]}