from collections import OrderedDict

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from data.utils_experiment.experiment import Experiment
from emulator.emulator_with_search import EmulatorWithSearch
from plot.workflow import fit
from projects.simple_paper.search_strategy import SearchStrategy, get_params_emulator, get_params_search
from utils.utils_log import log_info


def main_split_comparison(fast: bool = False):
    n_iter = 1 if fast else 10
    model_selections = ['best', 'validated']
    validation_splits = [ValidationSplit.START, ValidationSplit.SYMMETRICAL, ValidationSplit.END]
    search_strategies = [SearchStrategy.TOP10_FULL_RANGE, SearchStrategy.ALL_FULL_RANGE][:1]
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
            emulator = EmulatorWithSearch(**params_emulator, **params_search)
            fit(emulator, dataset, refit=False)
            experiments.append(emulator.experiment_.experiment_path)
        validation_split_to_experiment_paths[validation_split] = experiments
    # Plot an array with the results for both the model_selection 'best' and 'validated'
    for model_selection in model_selections:
        validation_split_name_to_rmse_test_list = OrderedDict()
        for validation_split, experiment_paths in validation_split_to_experiment_paths.items():
            experiments = [Experiment(experiment_path, model_selection) for experiment_path in experiment_paths]
            rmse_test_list = [experiment for experiment in experiments]
            validation_split_name_to_rmse_test_list[str(validation_split)] = rmse_test_list
        pass

if __name__ == '__main__':
    main_split_comparison(fast=False)