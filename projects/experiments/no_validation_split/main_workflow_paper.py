from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from plot.workflow import workflow
from projects.experiments.split_comparison.utils_params import get_params_search


def main_workflow_root_random_hyperparameter_settings(n_iter: int):
    params_emulator =  {
        "maxsize": 20,
        "optimizer_f_calls_limit": 10000,
        "population_size": 31,
        # "unary_operators": ["square"]
    }
    param_search = get_params_search(n_iter, scaling_factor=1.)
    workflow(dataset_npp_season_v1, params_emulator, param_search)


if __name__ == '__main__':
    main_workflow_root_random_hyperparameter_settings(n_iter=1)

