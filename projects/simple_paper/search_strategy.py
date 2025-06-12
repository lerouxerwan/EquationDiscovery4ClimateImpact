from enum import StrEnum
from typing import Any

from projects.simple_paper.utils_hyperparameters import get_param_name_to_values


class SearchStrategy(StrEnum):
    ALL_FULL_RANGE = 'all hyperparameter full range'

def get_params_emulator(search_strategy: SearchStrategy) -> dict[str, Any]:
    if search_strategy in [SearchStrategy.ALL_FULL_RANGE]:
        return {}
    else:
        raise NotImplementedError

def get_params_search(search_strategy: SearchStrategy) -> dict[str, Any]:
    # Full range strategies
    if search_strategy in [SearchStrategy.ALL_FULL_RANGE]:
        return {'param_grid': get_param_name_to_values(), 'search_style': 'random'}
    else:
        raise NotImplementedError




