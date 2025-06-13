from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_plot import plot_diagnosis
from plot.workflow import workflow, fit
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test
from projects.simple_paper.utils_hyperparameters import get_param_name_to_values


def main_analyze_equation():
    param_name = 'weight_insert_node'
    param_grid = {param_name: get_param_name_to_values()[param_name]}
    params_emulator = {'model_selection': 'validated'}
    params_search = {'param_grid': param_grid, 'search_style': 'grid'}
    emulator = EmulatorWithSearch(**params_emulator, **params_search)
    fit(emulator, get_dataset_preliminary_test())
    #  Generate diagnosis plot for the fit
    plot_diagnosis(emulator, dataset_npp_season_v1)

if __name__ == '__main__':
    main_analyze_equation()