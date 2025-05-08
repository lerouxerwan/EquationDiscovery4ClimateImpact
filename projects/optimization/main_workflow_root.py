from data.utils_dataset.npp_season_v1 import dataset_values_npp_season_v1
from projects.optimization.utils_worflow import workflow


def main_workflow_root_one_hyperparameter_setting():
    params_emulator = {
        "model_selection": "custom",
        "niterations": 2,
    }
    params_search = {
        "n_iter": 6,
        "scaling_factor": 2,
        "n_jobs": -1,
    }
    workflow(dataset_values_npp_season_v1, params_emulator, params_search)


def main_workflow_root_random_hyperparameter_settings():
    params_emulator = {
        "maxsize": 30,
        "model_selection": "custom",
        "unary_operators": ["square", "sqrt"],
        "binary_operators": ["+", "-", "*", "/"],
        "population_size": 31,
        "topn": 2,
        "optimizer_nrestarts": 1,
        "optimizer_iterations": 3,
        'fraction_replaced_hof': 0.25,
        "adaptive_parsimony_scaling": 300.,
        "optimizer_f_calls_limit": 10000,
        "weight_rotate_tree": 3.,
        "tournament_selection_n": 7,
        "tournament_selection_n": 7,
        "ncycles_per_iteration": 100,
        "niterations": 140,
    }
    params_search = {
        "n_iter": 1000,
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
    workflow(dataset_values_npp_season_v1, params_emulator, params_search)


if __name__ == '__main__':
    # main_workflow_root_one_hyperparameter_setting()
    main_workflow_root_random_hyperparameter_settings()

