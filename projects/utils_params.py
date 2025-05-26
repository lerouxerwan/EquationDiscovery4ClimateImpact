from utils.utils_run import MAX_NB_JOBS


def get_params_emulator():
    return {
        "maxsize": 20,
        "optimizer_f_calls_limit": 10000,
        "population_size": 31,
        "unary_operators": ["square"]
    }

def get_params_search(n_iter: int):
    return {
        "n_iter": n_iter,
        "n_jobs": MAX_NB_JOBS,
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