from enum import StrEnum
from typing import Any

from projects.simple_paper.utils_hyperparameters import get_param_name_to_values


class SearchStrategy(StrEnum):
    ALL_FULL_RANGE = 'all hyperparameter full range'
    TOP10_FULL_RANGE = 'top 10 hyperparameter full range'

def get_params_emulator(search_strategy: SearchStrategy) -> dict[str, Any]:
    if search_strategy in [SearchStrategy.ALL_FULL_RANGE, SearchStrategy.TOP10_FULL_RANGE]:
        return {}
    else:
        raise NotImplementedError

def get_params_search(search_strategy: SearchStrategy) -> dict[str, Any]:
    if search_strategy in [SearchStrategy.ALL_FULL_RANGE, SearchStrategy.TOP10_FULL_RANGE]:
        #  Full range strategies
        param_grid = get_param_name_to_values()
        if search_strategy is SearchStrategy.TOP10_FULL_RANGE:
            param_grid = {param_name: param_value for param_name, param_value in param_grid.items()
                          if param_name in top10_param_names}
        return {'param_grid': param_grid, 'search_style': 'random'}
    else:
        raise NotImplementedError


ordered_param_names = ['weight insert node', 'weight delete node', 'populations', 'fraction replaced hof', 'weight simplify', 'weight randomize', 'weight rotate tree', 'weight mutate operator', 'weight add node', 'tournament selection n', 'topn', 'perturbation factor', 'weight do nothing', 'unary operators', 'population size', 'maxsize', 'weight swap operands', 'ncycles per iteration', 'weight mutate constant', 'tournament selection p', 'weight optimize', 'warmup maxsize by', 'optimize probability', 'crossover probability', 'fraction replaced', 'adaptive parsimony scaling', 'niterations', 'probability negate constant', 'optimizer f calls limit']
top10_param_names = ordered_param_names[:10]




