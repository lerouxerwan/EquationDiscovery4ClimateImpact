from projects.paper.utils_paper import filename_dataset_paper
from projects.utils_workflow_root import workflow_root


def main_workflow(fast: bool = False):
    if fast:
        params_emulator = {
            # "select_k_features": 3,
            "n_iter": 1,
            "scaling_factor": 0,
            # "unary_operators": ["square", "sqrt"],
            "model_selection": "custom",

            # "param_grid": {'populations': [15, 20]},
            # "feature_selection_name": 'PySRDefault',
            # "niterations": 5,
            # "remove_duplicate_features": True,
        }
    else:
        params_emulator = {
            # "select_k_features": 5,
            # "remove_duplicate_features": True,
            # "duplicate_feature_threshold": 0.6,
            "n_iter": 500,
            "population_size": 31,
            # 'niterations': 100,
            "unary_operators": ["square", "sqrt"],
            "binary_operators": ["+", "*", "/", "-"],
            # Hyperparameter to optimize around (/2, x2) their default or specified value
            "scaling_factor": 2,
            # "populations": 10,
            # "optimize_probability": 0.5,
            "param_list_to_optimize_around_default": [
                "niterations",
                "adaptive_parsimony_scaling",
                "fraction_replaced_hof",
                "populations",
                "population_size"
            ]
            # "param_list_to_optimize_around_default": ['adaptive_parsimony_scaling',
            #                                           'fraction_replaced_hof',
            #                                           'weight_mutate_constant',
            #                                           "optimize_probability",
            #                                           "optimizer_iterations",
            #                                           "perturbation_factor"],
        }
    # Run workflow for several number of features
    workflow_root(filename_dataset_paper, **params_emulator)


if __name__ == '__main__':
    fast = True
    main_workflow(fast)


