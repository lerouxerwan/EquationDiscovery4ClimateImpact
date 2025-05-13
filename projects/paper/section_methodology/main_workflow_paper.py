from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from projects.optimization.utils_worflow import workflow


def main_workflow_root_random_hyperparameter_settings(n_iter: int):
    params_emulator = {
        "maxsize": 20,
        "optimizer_f_calls_limit": 10000,
        "population_size": 31,
        "unary_operators": ["square", "sqrt"]
    }
    params_search = {
        "n_iter": n_iter,
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
    workflow(dataset_npp_season_v1, params_emulator, params_search)


if __name__ == '__main__':
    main_workflow_root_random_hyperparameter_settings(n_iter=1000)

