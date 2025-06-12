from typing import OrderedDict

import pandas as pd

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.workflow import fit
from projects.experiments.preliminary.utils_dataset_preliminary import get_dataset_preliminary_test
from projects.simple_paper.utils_hyperparameters import get_param_name_to_values, relative_error
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info


def main_hyperparameter_sensitivity(fast: bool = False):
    model_selection = 'validated'
    niterations = 2 if fast else 100
    dataset = get_dataset_preliminary_test()
    # Compute rmse for the default hyperparameter
    emulator_default_params = Emulator(model_selection, niterations=niterations)
    # fit(emulator_default_params, dataset)
    # rmse_validation = emulator_default_params.compute_loss_for_set(dataset.X_train, dataset.y_train,
    #                                                                dataset.validation_mask, True, Metric.RMSE)
    # print(rmse_validation)
    rmse_validation = 1.7392657669045026
    # For each parameter we compute a row containing the minimum rmse reached for all the tested values
    rows = []
    for param_name, param_values in get_param_name_to_values().items():
        if fast:
            param_values = param_values[:1] + param_values[-1:]
        log_info('\n')
        log_info(f'Run/Load grid search with "{param_name}" using {len(param_values)} values')
        # Fit a grid search
        param_grid = {param_name: param_values}
        emulator_with_search = EmulatorWithSearch(model_selection, niterations=niterations, **{'param_grid': param_grid, 'search_style': 'grid'})
        fit(emulator_with_search, dataset, refit=False)
        experiment = emulator_with_search.experiment_
        # Create a row of the future dataframe
        d = OrderedDict()
        d['Parameter'] = param_name.replace('_', ' ')
        d['Default value'] = emulator_default_params.get_params()[param_name]
        d['Best value from grid search'] = experiment.top_params[param_name]
        d['Default RMSE RCP4.5'] = rmse_validation
        d['Best RMSE RCP4.5 from grid search'] = experiment.top_rmse_validation
        gap_column_name = 'Difference in RMSE RCP4.5 (\%)'
        d[gap_column_name] = relative_error(rmse_validation, experiment.top_rmse_validation)
        # d['Optimized complexity'] = experiment.top_complexity
        rows.append(pd.Series(d))
    df = pd.concat(rows, axis=1).transpose()
    df = df.sort_values(by=gap_column_name)
    print(df.head())
    print_df_latex(df)
    print(df['Parameter'].to_list())

# ['weight insert node', 'weight delete node', 'populations', 'fraction replaced hof', 'weight simplify', 'weight randomize', 'weight rotate tree', 'weight mutate operator', 'weight add node', 'tournament selection n', 'topn', 'perturbation factor', 'weight do nothing', 'unary operators', 'population size', 'maxsize', 'weight swap operands', 'ncycles per iteration', 'weight mutate constant', 'tournament selection p', 'weight optimize', 'warmup maxsize by', 'optimize probability', 'crossover probability', 'fraction replaced', 'adaptive parsimony scaling', 'niterations', 'probability negate constant', 'optimizer f calls limit']



if __name__ == '__main__':
    main_hyperparameter_sensitivity(fast=False)