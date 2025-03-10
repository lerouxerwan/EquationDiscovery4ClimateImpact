import math
from typing import Optional, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def get_param_grid(estimator: BaseEstimator, scaling_factor: float, search_cv_type: type, n_iter: int,
                   param_list_to_optimize: Optional[list[str]] = None) -> dict[str, Any]:
    """Create param_grid based on a list 'param_list_to_optimize' that specifies the list of hyperparameters to optimize
    Each hyperparameter will be optimized by testing values around the value specified for this hyperparameter
    If search_cv_type is:
        -a random search, then hyperparameters are sampled between [value / scaling_factor, value * scaling_factor]
        - a grid search, then hyperparameters are evenly spaced in [value / scaling_factor, value * scaling_factor]"""
    assert (param_list_to_optimize is None) or isinstance(param_list_to_optimize, list)
    assert search_cv_type in [GridSearchCV, RandomizedSearchCV]
    #  By default, we consider a list with 1 hyperparameter: the number of iterations
    if param_list_to_optimize is None:
        param_list_to_optimize = ['niterations']
    param_grid = dict()
    for key in param_list_to_optimize:
        value = estimator.__getattribute__(key)
        min_value = value if scaling_factor == 0 else value / scaling_factor
        max_value = value if scaling_factor == 0 else value * scaling_factor
        if isinstance(value, int):
            min_value = math.ceil(min_value)
        param_grid[key] = [min_value, max_value]
    # For grid search, hyperparameters are evenly spaced (on a log scale)
    if search_cv_type is GridSearchCV:
        param_grid = {k: np.geomspace(min_value, max_value, n_iter) for k, (min_value, max_value) in param_grid.items()}
    return param_grid
