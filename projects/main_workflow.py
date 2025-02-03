from projects.utils_workflow import workflow
from utils.utils_run import NB_CORES

FILENAME = r"NPP_season.csv"


def main_workflow(fast: bool = False):
    if fast:
        params_emulator = {
            "select_k_features": 3,
            "n_iter": 2,
            "n_jobs": 1,
            "param_grid": {'populations': [10, 20]},
            "feature_selection_name": 'PySRDefault',
        }
    else:
        params_emulator = {
            "select_k_features": 5,
            "n_iter": 200,
            "n_jobs": 1,
            "population_size": 31,
            "maxsize": 20,
            "unary_operators": ["exp", "log", "square", "sqrt"],
            "binary_operators": ["+", "*", "/", "-"],
            # Hyperparameter to optimize around (/2, x2) their default or specified value
            "scaling_factor": 2,
            'niterations': 500,
            "param_list_to_optimize_around_default": ['niterations', 'adaptive_parsimony_scaling',
                                                      'fraction_replaced_hof', 'populations',
                                                      'population_size'],
            "feature_selection_name": ['PySRDefault', 'ExpertKnowledgeSeason'][0],
        }
    # Run workflow for several number of features
    workflow(FILENAME, **params_emulator)


if __name__ == '__main__':
    fast = True
    main_workflow(fast)


