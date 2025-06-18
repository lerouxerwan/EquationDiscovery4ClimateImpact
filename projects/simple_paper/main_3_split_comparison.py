from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn import clone

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from data.utils_experiment.experiment import Experiment
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_metric.metric import Metric
from plot.workflow import fit
from projects.simple_paper.search_strategy import SearchStrategy, get_params_emulator, get_params_search
from utils.utils_latex import print_df_latex
from utils.utils_log import log_info


def main_split_comparison(fast: bool = False):
    n_iter = 10 if fast else 25
    model_selections = ['best', 'validated']
    validation_splits = [ValidationSplit.RCP_START, ValidationSplit.EXTREME, ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END][:]
    search_strategies = [SearchStrategy.TOP5_VALIDATED, SearchStrategy.TOP5_BEST][:]
    # Run emulator fit (and get experiment) for every validation_splits and every search_strategies
    validation_split_to_experiment_paths = OrderedDict()
    for validation_split in validation_splits:
        experiments = []
        for search_strategy in search_strategies:
            log_info(f'Run for validation_split={validation_split} & search_strategy={search_strategy}')
            dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.3, validation_split)
            params_emulator = get_params_emulator(search_strategy)
            params_search = get_params_search(search_strategy)
            params_search['n_iter'] = n_iter
            params_search['n_jobs'] = -1
            emulator = EmulatorWithSearch(**params_emulator, **params_search)
            fit(emulator, dataset, refit=False)
            compute_and_save_test_rmse(emulator, dataset, model_selections)
            experiments.append(emulator.experiment_.experiment_path)
        validation_split_to_experiment_paths[validation_split] = experiments
    # Plot an array with the results for both the model_selection 'best' and 'validated'
    for model_selection in model_selections:
        validation_split_name_to_rmse_test_list = OrderedDict()
        for validation_split, experiment_paths in validation_split_to_experiment_paths.items():
            experiments = [Experiment(experiment_path, model_selection) for experiment_path in experiment_paths]
            log_info(f'{model_selection}, {validation_split}')
            rmse_test_list = [experiment.top_rmse_test for experiment in experiments]
            print(model_selection, validation_split, [experiment.top_expr for experiment in experiments], rmse_test_list)
            validation_split_name_to_rmse_test_list[str(validation_split)] = rmse_test_list
        df = pd.DataFrame(index=[str(s) for s in search_strategies], data=validation_split_name_to_rmse_test_list)
        print(f'\nRESULTS for {model_selection} model_selection, and n_iter={n_iter}\n')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Sampling strategy'}, inplace=True)
        print_df_latex(df)


def compute_and_save_test_rmse(emulator: Emulator, dataset: Dataset, model_selections: list[str]):
    try:
        _ = emulator.experiment_.top_rmse_test
    except KeyError:
        _compute_and_save_test_rmse(emulator, dataset, model_selections)

def _compute_and_save_test_rmse(emulator: Emulator, dataset: Dataset, model_selections: list[str]):
    log_info('Compute and save test rmse')
    experiment = emulator.experiment_
    df = experiment.df_cv_results.copy()
    emulator_params = emulator.get_params()
    # Add test rmse to the dataframe
    for model_selection in model_selections:
        # Set params and model selection
        emulator.set_params(**emulator_params)
        emulator.set_model_selection(model_selection)
        # Fill
        index_name = emulator.experiment_.df_cv_results.index[0]
        column_name = emulator.experiment_.rmse_test_column_name
        df[column_name] = np.nan
        log_info(f'Compute {column_name} for index={index_name}')
        #  Refit with the top params associated to the model_selected
        fit(emulator, dataset)
        df.loc[index_name, column_name] = emulator.compute_loss(dataset.X_test, dataset.y_test, Metric.RMSE)
    # Save dataframe to file
    emulator.set_params(**emulator_params)
    emulator.experiment_.save_search_results(df, emulator.non_default_params)

if __name__ == '__main__':
    main_split_comparison(fast=False)