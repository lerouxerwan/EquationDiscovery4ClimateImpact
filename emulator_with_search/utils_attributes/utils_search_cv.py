from typing import Optional

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from emulator_with_search.utils_param_grid.utils_params_distribution import get_param_distributions
from utils.utils_run import random_seed


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
