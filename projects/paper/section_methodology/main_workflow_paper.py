from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from projects.optimization.utils_worflow import workflow
from projects.utils_params import get_params_emulator, get_params_search


def main_workflow_root_random_hyperparameter_settings(n_iter: int):
    workflow(dataset_npp_season_v1, get_params_emulator(), get_params_search(n_iter))


if __name__ == '__main__':
    main_workflow_root_random_hyperparameter_settings(n_iter=100)

