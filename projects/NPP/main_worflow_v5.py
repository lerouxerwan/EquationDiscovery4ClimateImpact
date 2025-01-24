from projects.utils_workflow import workflow
from utils.utils_run import NB_CORES

FILENAME = r"v5_NPPz_annual_season_GOL4_allDepths_HIST_20_RCP85_94_RCP45_94_all_25_month_season.csv"


def main_workflow(fast: bool = False):
    if fast:
        nb_features_list = [3]
        params_emulator = {
            "n_iter": 2,
            "n_jobs": 1,
            "param_grid": {'populations': [10, 20]},
        }
    else:
        nb_features_list = [4]
        params_emulator = {
            "n_iter": 100,
            "n_jobs": 1,
            "population_size": 31,
            "maxsize": 15,
            "unary_operators": ["exp", "log", "square", "sqrt"],
            "binary_operators": ["+", "*", "/", "-"],
            # Hyperparameter to optimize around (/2, x2) their default or specified value
            "scaling_factor": 2,
            "adaptive_parsimony_scaling": 2000.,
            "niterations": 100.,
            "param_list_to_optimize_around_default": ['niterations', 'adaptive_parsimony_scaling',
                                                      'fraction_replaced_hof', 'populations',
                                                      'population_size'],
            "feature_selection_name": 'ExpertKnowledge',
        }
    # Run workflow for several number of features
    for nb_features in nb_features_list:
        workflow(FILENAME, nb_features, **params_emulator)


if __name__ == '__main__':
    fast = False
    main_workflow(fast)


