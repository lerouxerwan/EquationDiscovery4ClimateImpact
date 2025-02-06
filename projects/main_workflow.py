from projects.utils_workflow import workflow
from utils.utils_run import NB_CORES

FILENAME = r"NPP_season.csv"


def main_workflow(fast: bool = False):
    if fast:
        params_emulator = {
            "select_k_features": 3,
            "n_iter": 2,
            "param_grid": {'populations': [10, 20]},
            "feature_selection_name": 'PySRDefault',
        }
    else:
        params_emulator = {
            "select_k_features": 7,
            "n_iter": 500,
            "population_size": 31,
            "niterations": 10,
            "maxsize": 20,
            "unary_operators": ["exp", "log", "square", "sqrt"],
            "binary_operators": ["+", "*", "/", "-"],
            # Hyperparameter to optimize around (/2, x2) their default or specified value
            "scaling_factor": 2,
            "param_list_to_optimize_around_default": ['niterations', 'adaptive_parsimony_scaling',
                                                      'fraction_replaced_hof', 'populations',
                                                      'population_size'],
        }
    # Run workflow for several number of features
    workflow(FILENAME, **params_emulator)


if __name__ == '__main__':
    fast = False
    main_workflow(fast)


