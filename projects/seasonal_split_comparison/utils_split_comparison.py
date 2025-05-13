from typing import Any, Optional

from data.utils_dataset.dataset import Dataset
from emulator.utils_metric.utils_metric_function import root_mean_squared_error
from emulator_with_search.pysr_emulator_with_search import PySREmulatorWithSearch


def get_rmse_test(dataset: Dataset, params_emulator: dict[str, Any], params_search: Optional[dict[str, Any]] = None) -> float:
    """Load test RMSE"""
    # return 2.0
    emulator = PySREmulatorWithSearch(**params_emulator, **params_search)
    emulator.fit(dataset.X_train, dataset.y_train, variable_names=dataset.X_variables_names, X_units=dataset.X_units,
                 y_units=dataset.y_units, validation_mask=dataset.validation_mask)
    y_test_predict = emulator.predict(dataset.X_test)
    return root_mean_squared_error(dataset.y_test, y_test_predict)

def get_params_emulator():
    return {
        "maxsize": 20,
        "optimizer_f_calls_limit": 10000,
        "population_size": 31,
        "unary_operators": ["square"]
    }

def get_params_search(fast: True):
    return {
        "n_iter": 10 if fast else 100,
        "n_jobs": -1,
        "scaling_factor": 2.,
        'param_list_to_optimize': [
            "populations",
            "niterations",
            "fraction_replaced_hof",
            "adaptive_parsimony_scaling",
            "ncycles_per_iteration",
            "fraction_replaced",
            "weight_add_node",
            "weight_insert_node",
            "weight_delete_node",
            "weight_do_nothing",
            "weight_mutate_constant",
            "weight_mutate_operator",
            "weight_swap_operands",
            "weight_rotate_tree",
            "weight_randomize",
            "weight_simplify",
            "crossover_probability",
            "topn",
            "optimizer_nrestarts",
            "optimizer_f_calls_limit",
            "optimize_probability",
            "perturbation_factor",
            "probability_negate_constant",
            "tournament_selection_n"
        ]
    }