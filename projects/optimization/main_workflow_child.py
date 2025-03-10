from sklearn.model_selection import GridSearchCV

from projects.optimization.utils_worflow import workflow
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_child():
    search_path_to_start_from = 'best'
    search_path_to_start_from = '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/RandomizedSearchCV_1_nit1_nit100_100_sca0'
    params_emulator_child = {
        "n_iter": 10,
        "scaling_factor": 10,
        "search_cv_type": GridSearchCV,
    }
    workflow(filename_dataset_paper, search_path_to_start_from, **params_emulator_child)


if __name__ == '__main__':
    main_workflow_child()


