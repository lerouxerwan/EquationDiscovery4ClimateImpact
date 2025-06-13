from typing import OrderedDict

import pandas as pd

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_metric.metric import Metric
from plot.workflow import fit
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test
from projects.simple_paper.utils_hyperparameters import get_param_name_to_values, relative_error
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info



# COLUMN NAMES
HYPERPARAMETER_NAME_COLUMN_NAME = 'Name'
DEFAULT_VALUE_COLUMN_NAME = 'Default value'
TOP_VALUE_COLUMN_NAME = 'Top value'
TOP_RMSE_COLUMN_NAME = 'Top RMSE'
RELATIVE_DIFFERENCE_COLUMN_NAME = 'Drop in RMSE (\%)'
DEFAULT_FIT_TIME = 'Default duration (s)'
TOP_FIT_TIME = 'Top duration (s)'
MAX_FIT_TIME = 'Max duration (s)'


def main_hyperparameter_sensitivity(model_selection: str = 'best', fast: bool = False):
    for model_selection in ['validated', 'best'][:]:
        df = compute_df(fast, model_selection)
        log_info(f'RESULTS FOR {model_selection}')
        print_df_latex(df)
        print('\n', df[HYPERPARAMETER_NAME_COLUMN_NAME].to_list())


def compute_df(fast, model_selection):
    dataset = get_dataset_preliminary_test()
    #  Compute rmse for the default hyperparameter
    emulator_default_params = Emulator(model_selection)
    # fit(emulator_default_params, dataset)
    # rmse_validation = emulator_default_params.compute_loss_for_set(dataset.X_train, dataset.y_train,
    #                                                                dataset.validation_mask, True, Metric.RMSE)
    # print(rmse_validation)
    model_selection_to_rmse_validation = {
        'validated': 1.7392657669045026,
        'best': 1.8087302251690016,
    }
    rmse_validation = model_selection_to_rmse_validation[model_selection]
    #  For each parameter we compute a row containing the minimum rmse reached for all the tested values
    rows = []
    param_name_to_values = get_param_name_to_values()
    if fast:
        param_name_to_values = {p: v for p, v in param_name_to_values.items() if p in ['weight_insert_node']}
    for param_name, param_values in param_name_to_values.items():
        log_info('\n')
        log_info(f'Run/Load grid search with "{param_name}" using {len(param_values)} values')
        #  Fit a grid search
        param_grid = {param_name: param_values}
        emulator_with_search = EmulatorWithSearch(model_selection, **{'param_grid': param_grid, 'search_style': 'grid'})
        fit(emulator_with_search, dataset, refit=False)
        experiment = emulator_with_search.experiment_
        #  
        #  Create a row of the future dataframe
        d = OrderedDict()
        d[HYPERPARAMETER_NAME_COLUMN_NAME] = param_name.replace('_', ' ')
        d[DEFAULT_VALUE_COLUMN_NAME] = emulator_default_params.get_params()[param_name]
        d[TOP_VALUE_COLUMN_NAME] = experiment.top_params[param_name]
        d[TOP_RMSE_COLUMN_NAME] = experiment.top_rmse_validation  # rmse_validation correspond to RCP4.5 for this dataset
        d[RELATIVE_DIFFERENCE_COLUMN_NAME] = relative_error(rmse_validation, experiment.top_rmse_validation)
        d[TOP_FIT_TIME] = experiment.top_fit_time
        d[MAX_FIT_TIME] = experiment.max_fit_time  #
        rows.append(pd.Series(d))
    df = pd.concat(rows, axis=1).transpose()
    df = df.sort_values(by=RELATIVE_DIFFERENCE_COLUMN_NAME)
    return df



if __name__ == '__main__':
        main_hyperparameter_sensitivity(fast=False)