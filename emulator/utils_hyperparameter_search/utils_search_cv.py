from typing import Optional, Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterSampler

from emulator.utils_hyperparameter_search.utils_params_distribution import get_param_distributions
from utils.utils_run import random_seed


def get_cv(validation_mask: np.ndarray):
    """Generator that returns the single split for the validation, i.e. train_indices, validation_indices"""
    indices = np.arange(len(validation_mask))
    yield indices, indices[validation_mask]


def get_search_cv_kwargs(search_cv_type: type, param_grid: dict, n_iter: Optional[int]) -> dict:
    """Additional arguments for the instantiation of search_cv object, depending on the type of search"""
    if issubclass(search_cv_type, GridSearchCV):
        return {'param_grid': param_grid}
    elif issubclass(search_cv_type, RandomizedSearchCV):
        assert isinstance(param_grid, dict)
        assert n_iter is not None
        return {'n_iter': n_iter, 'random_state': random_seed,
                'param_distributions': get_param_distributions(param_grid)}
    else:
        raise NotImplementedError

def get_random_params_list_from_param_grid(param_grid: dict, n_iter: Optional[int]) -> list[dict[str, Any]]:
    """Return the list of hyperparameters sampled for the hyperparameter search"""
    return list(ParameterSampler(**get_search_cv_kwargs(RandomizedSearchCV, param_grid, n_iter)))

