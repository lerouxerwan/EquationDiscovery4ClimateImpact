from projects.optimization.utils_worflow import workflow
from projects.optimization.utils_workflow_child import workflow_child
from projects.paper.utils_paper import filename_dataset_paper


def main_workflow_child():
    search_path_to_start_from = 'best'
    # search_path_to_start_from = '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/random_1_nit100_100_sca0'
    params_emulator_child = {
        "n_iter": 2,
        "scaling_factor": 1.2,
        'param_list_to_optimize': ['populations', 'niterations', 'fraction_replaced_hof']
    }
    workflow_child(filename_dataset_paper, search_path_to_start_from, **params_emulator_child)

def main_workflow_child_marginal_analysis():
    search_path_to_start_from = '/home/e23lerou/Documents/EquationDiscovery4ClimateImpact/data/search/6624636351014803864/random_1_nit100_100_sca0'
    params_emulator_child = {
        "n_iter": 10,
        "scaling_factor": 2,
        "search_style": "grid",
    }
    # 'adaptive_parsimony_scaling', 'fraction_replaced_hof', 'crossover_probability',
    for param_name in ['maxsize', 'ncycles_per_iteration', 'population_size', 'populations']:
        params_emulator_child['param_list_to_optimize'] = [param_name]
        workflow_child(filename_dataset_paper, search_path_to_start_from, **params_emulator_child)


if __name__ == '__main__':
    main_workflow_child()
    # main_workflow_child_marginal_analysis()


