

from typing import OrderedDict

import pandas as pd

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from emulator.utils_hyperparameter_search.utils_column_names import get_cv_results_column_name, \
    RMSE_VALIDATION_COLUMN_NAME
from plot.utils_metric.metric import Metric
from plot.workflow import fit
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test
from projects.simple_paper.main_2_hyperparameter_sensitivity import HYPERPARAMETER_NAME_COLUMN_NAME
from projects.simple_paper.utils_hyperparameters import get_param_name_to_values, relative_error
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info



# COLUMN NAMES
NB_PARAMETER_VALUES_COLUMN_NAME = 'Number of values'
PERCENTAGE_GOOD_SELECTION_COLUMN_NAME = 'Selected equation is the most adjusted to RCP4.5 (\%)'
MEAN_DROP_RMSE_FOR_NON_OPTIMAL_SELECTION_COLUMN_NAME = 'Mean drop in RMSE, between selected equation and the most adjusted (\%)'

column_name_to_format = {
    HYPERPARAMETER_NAME_COLUMN_NAME: 'p{40mm}',
    NB_PARAMETER_VALUES_COLUMN_NAME: 'p{16mm}',
    PERCENTAGE_GOOD_SELECTION_COLUMN_NAME: 'p{42mm}',
    MEAN_DROP_RMSE_FOR_NON_OPTIMAL_SELECTION_COLUMN_NAME: 'p{54mm}',
}

def main_hyperparameter_sensitivity(fast: bool = False):


    dataset = get_dataset_preliminary_test()
    rows = []
    param_name_to_values = get_param_name_to_values()
    if fast:
        param_name_to_values = {p: v for p, v in param_name_to_values.items() if p in ['weight_insert_node']}
    for param_name, param_values in param_name_to_values.items():
        log_info('\n')
        log_info(f'Run/Load grid search with "{param_name}" using {len(param_values)} values')
        #  Fit a grid search
        param_grid = {param_name: param_values}
        emulator_with_search = EmulatorWithSearch(**{'param_grid': param_grid, 'search_style': 'grid', 'n_jobs': -1})
        fit(emulator_with_search, dataset, refit=False)
        experiment = emulator_with_search.experiment_
        #  
        #  Create a row of the future dataframe
        d = OrderedDict()
        d[HYPERPARAMETER_NAME_COLUMN_NAME] = param_name.replace('_', ' ')
        d[NB_PARAMETER_VALUES_COLUMN_NAME] = len(experiment.df_cv_results)
        d[PERCENTAGE_GOOD_SELECTION_COLUMN_NAME] = experiment.percentage_of_best_same_as_validated
        d[MEAN_DROP_RMSE_FOR_NON_OPTIMAL_SELECTION_COLUMN_NAME] = experiment.mean_difference_in_rmse_validation_for_best_not_same_as_validated
        rows.append(pd.Series(d))
    df = pd.concat(rows, axis=1).transpose()
    df = df.sort_values(by=MEAN_DROP_RMSE_FOR_NON_OPTIMAL_SELECTION_COLUMN_NAME)
    print_df_latex(df, column_format=''.join([column_name_to_format[c] for c in df.columns]))

if __name__ == '__main__':
        main_hyperparameter_sensitivity(fast=False)