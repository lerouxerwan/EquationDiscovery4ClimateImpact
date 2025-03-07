from emulator_with_search.utils_workflow.utils_worflow import workflow
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_child():
    search_path_to_start_from = '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/RandomizedSearchCV_1_nit1_nitons10_ada1040.0_1040.0_fra0.1_0.1_nit10_10_sca0'
    params_emulator_child = {
        "n_iter": 10,
        "scaling_factor": 2,
    }
    # search_path_to_start_from = '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/RandomizedSearchCV_10_fra0.06140000000000002_nit10_ada520.0_2080.0_fra0.0_0.1_nit5_20_sca2_thr1.0005'
    # params_emulator_child = {}
    workflow(filename_dataset_paper, search_path_to_start_from, **params_emulator_child)


if __name__ == '__main__':
    main_workflow_child()


