from enum import StrEnum
from typing import Any

from projects.simple_paper.utils_hyperparameters import get_param_name_to_values


class SearchStrategy(StrEnum):
    ALL = 'all hyperparameters'
    ALL_EXCLUDING_3 = 'all hyperparameters except three'
    TOP5_BEST = 'top 5 hyperparameters for equation selected by default'
    TOP5_VALIDATED = 'top 5 hyperparameters for equation most adjusted to RCP4.5'

def get_params_emulator(search_strategy: SearchStrategy) -> dict[str, Any]:
    return {}
    # if search_strategy in [SearchStrategy.ALL_EXCLUDING_3]:
    #     return {}
    # else:
    #     raise NotImplementedError

def get_params_search(search_strategy: SearchStrategy) -> dict[str, Any]:
    # Compute the list of param_names to take into account
    if search_strategy is SearchStrategy.ALL_EXCLUDING_3:
        raise NotImplementedError
    elif search_strategy is SearchStrategy.TOP5_VALIDATED:
        selected_param_names = get_ordered_param_names('validated')[:5]
    elif search_strategy is SearchStrategy.TOP5_BEST:
        selected_param_names = get_ordered_param_names('best')[:5]
    else:
        raise NotImplementedError
    param_grid = get_param_name_to_values()
    param_grid = {param_name: param_value for param_name, param_value in param_grid.items()
                  if param_name in selected_param_names}
    return {'param_grid': param_grid, 'search_style': 'random'}



long_param_names = {'populations', 'ncycles_per_iteration', 'niterations', 'weight_optimize', 'maxsize'}

def get_ordered_param_names_without_long_params(model_selection: str) -> list[str]:
    ordered_param_names = get_ordered_param_names(model_selection)
    return [param_name for param_name in ordered_param_names if param_name not in long_param_names]

def get_ordered_param_names(model_selection: str) -> list[str]:
    if model_selection == 'validated':
        return ['optimize_probability', 'weight_insert_node', 'weight_mutate_operator', 'weight_swap_operands', 'weight_randomize', 'niterations', 'tournament_selection_p', 'weight_do_nothing', 'tournament_selection_n', 'perturbation_factor', 'ncycles_per_iteration', 'maxsize', 'population_size', 'unary_operators', 'populations', 'weight_optimize', 'weight_mutate_constant', 'weight_simplify', 'fraction_replaced_hof', 'weight_delete_node', 'weight_add_node', 'adaptive_parsimony_scaling', 'probability_negate_constant', 'crossover_probability', 'warmup_maxsize_by', 'weight_rotate_tree', 'fraction_replaced', 'topn', 'optimizer_f_calls_limit']
    elif model_selection == "best":
        return ['weight_insert_node', 'weight_optimize', 'weight_delete_node', 'populations', 'fraction_replaced_hof', 'weight_simplify', 'weight_randomize', 'optimize_probability', 'weight_rotate_tree', 'weight_mutate_operator', 'tournament_selection_p', 'weight_add_node', 'topn', 'perturbation_factor', 'tournament_selection_n', 'weight_do_nothing', 'unary_operators', 'population_size', 'maxsize', 'weight_swap_operands', 'ncycles_per_iteration', 'weight_mutate_constant', 'warmup_maxsize_by', 'crossover_probability', 'fraction_replaced', 'adaptive_parsimony_scaling', 'niterations', 'probability_negate_constant', 'optimizer_f_calls_limit']
    else:
        raise NotImplementedError



