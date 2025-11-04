from enum import StrEnum
from typing import Optional, Any
from optimization.utils_params.utils_default_centred_values import get_param_name_to_default_centred_values


class ParamNameToValues(StrEnum):
    DEFAULT_CENTRED = 'default_centred'
    DEFAULT_CENTRED_WO_OPERATORS = 'default_centred_wo_operators'

def get_param_name_to_values(param_name_to_values: ParamNameToValues) -> Optional[dict[str, list[Any]]]:
    if param_name_to_values in [ParamNameToValues.DEFAULT_CENTRED, ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS]:
        d = get_param_name_to_default_centred_values()
    else:
        raise ValueError(param_name_to_values)
    if isinstance(d, dict):
        param_names_defined = set(d.keys())
        param_names_expected = set(param_names)
        assert param_names_defined == param_names_expected, \
            f'Undefined values for {param_names_expected - param_names_defined}'
    if param_name_to_values is ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS:
        for param_name in ['unary_operators', 'binary_operators']:
            if param_name in d:
                d.pop(param_name)
    return d

param_names = ['maxsize', 'warmup_maxsize_by', 'unary_operators', 'populations', 'population_size', 'ncycles_per_iteration',
     'topn', 'optimizer_f_calls_limit', 'optimize_probability', 'tournament_selection_p', 'tournament_selection_n',
     'weight_optimize', 'adaptive_parsimony_scaling', 'fraction_replaced', 'fraction_replaced_hof', 'weight_add_node',
     'weight_insert_node', 'weight_delete_node', 'weight_do_nothing', 'weight_mutate_constant',
     'weight_mutate_operator', 'weight_swap_operands', 'weight_rotate_tree', 'weight_randomize', 'weight_simplify',
     'crossover_probability', 'perturbation_factor', 'probability_negate_constant', 'maxdepth', 'niterations']

