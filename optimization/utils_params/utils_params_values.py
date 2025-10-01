from enum import StrEnum
from typing import Optional, Any
from optimization.utils_params.utils_default_centred_values import get_param_name_to_default_centred_values


class ParamsValues(StrEnum):
    DEFAULT_CENTRED = 'default_centred'

def get_param_name_to_values(params_values: Optional[ParamsValues]) -> Optional[dict[str, list[Any]]]:
    if params_values is None:
        param_name_to_values = None
    elif params_values is ParamsValues.DEFAULT_CENTRED:
        param_name_to_values = get_param_name_to_default_centred_values()
    else:
        raise ValueError(params_values)
    if isinstance(param_name_to_values, dict):
        param_names_defined = set(param_name_to_values.keys())
        param_names_expected = set(param_names)
        assert param_names_defined == param_names_expected, \
            f'Undefined values for {param_names_expected - param_names_defined}'
    return param_name_to_values

param_names = ['maxsize', 'warmup_maxsize_by', 'unary_operators', 'populations', 'population_size', 'ncycles_per_iteration',
     'topn', 'optimizer_f_calls_limit', 'optimize_probability', 'tournament_selection_p', 'tournament_selection_n',
     'weight_optimize', 'adaptive_parsimony_scaling', 'fraction_replaced', 'fraction_replaced_hof', 'weight_add_node',
     'weight_insert_node', 'weight_delete_node', 'weight_do_nothing', 'weight_mutate_constant',
     'weight_mutate_operator', 'weight_swap_operands', 'weight_rotate_tree', 'weight_randomize', 'weight_simplify',
     'crossover_probability', 'perturbation_factor', 'probability_negate_constant', 'maxdepth', 'niterations']

