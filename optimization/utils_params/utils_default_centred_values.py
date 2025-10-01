from itertools import combinations
from typing import Any

from emulator.emulator import Emulator

unary_operators = ['square', 'sqrt', 'exp', 'log']
factors = [2 * i for i in range(1, 6)]
up_and_down_factors= lambda v: [v / f for f in factors[::-1]] + [v * f for f in factors]
up_and_down_factors_int = lambda v: [int(f) for f in up_and_down_factors(v)]


def get_param_name_to_default_centred_values() -> dict[str, list[Any]]:
    emulator_with_default_params = Emulator()
    probability_values = [round(0.1 + i / 10, 1) for i in range(10)]
    nb_of_members = list(range(2, 21, 2))

    param_name_to_values = {
        "unary_operators": [[o] for o in unary_operators] + [list(c) for c in combinations(unary_operators, r=2)],
        "maxsize": list(range(13, 41, 3)),
        "warmup_maxsize_by": probability_values,
        'populations': [2] + up_and_down_factors_int(31)[1:], # avoid having 2 times the value "3" in the list
        'population_size': [3 * i + 17 for i in range(10)],
        'ncycles_per_iteration': up_and_down_factors_int(380),
        'topn': nb_of_members,
        'optimizer_f_calls_limit': up_and_down_factors_int(10_000),
        'optimize_probability': probability_values,
        'tournament_selection_p': probability_values,
        'tournament_selection_n': nb_of_members,
        'weight_optimize': probability_values,
        'maxdepth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'niterations': up_and_down_factors_int(1000),
    }


    # Add parameters with float values
    param_names_with_float_values = ['adaptive_parsimony_scaling',
                                     'fraction_replaced', 'fraction_replaced_hof',
                                     'weight_add_node', 'weight_insert_node', 'weight_delete_node',
                                     'weight_do_nothing', 'weight_mutate_constant', 'weight_mutate_operator',
                                     'weight_swap_operands', 'weight_rotate_tree', 'weight_randomize',
                                     'weight_simplify', 'crossover_probability',
                                     "perturbation_factor", "probability_negate_constant"]
    for param_name in param_names_with_float_values:
        default_param_value = emulator_with_default_params.get_params()[param_name]
        param_name_to_values[param_name] = up_and_down_factors(default_param_value)

    return param_name_to_values