from dataclasses import dataclass
from functools import cached_property
from random import sample
from typing import Optional, Any, OrderedDict

import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator_with_search import EmulatorWithSearch
from plot.utils_metric.metric import Metric
from plot.workflow import fit, compute_loss_test
from projects.simple_paper.utils_hyperparameters import get_param_name_to_values
from utils.utils_log import log_info


@dataclass
class ValidationWorkflow(object):
    validation_split: ValidationSplit
    n_iter: int
    nb_top_hyperparameters: int
    nb_hyperparameters: Optional[int] = None
    validation_size: float = 0.3

    def __post_init__(self):
        # Load dataset
        self.dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", self.validation_size, self.validation_split)
        # Load top emulator
        self.emulator = self.search_for_top_emulator()
        log_info('Fit top emulator')
        fit(self.emulator, self.dataset)
        # Compute rmse test
        self.rmse_test = compute_loss_test(self.emulator, self.dataset, Metric.RMSE)

    def search_for_top_emulator(self) -> EmulatorWithSearch:
        param_name_to_values = self.get_param_name_to_values()
        # Run marginal search
        log_info(f'Start marginal search with: {list(param_name_to_values.keys())}')
        param_name_to_emulator_with_search: dict[str, EmulatorWithSearch] = {}
        for param_name, param_values in param_name_to_values.items():
            log_info(f'Run marginal search for {param_name}')
            param_search = {'param_grid': {param_name: param_values}, 'search_style': 'grid', 'n_jobs': -1}
            emulator_with_search_marginal = EmulatorWithSearch(**self.params_emulator, **param_search)
            fit(emulator_with_search_marginal, self.dataset)
            param_name_to_emulator_with_search[param_name] = emulator_with_search_marginal
        # Select the top hyperparameters with respect to their validation loss
        key = lambda n: param_name_to_emulator_with_search[n].selected_validation_rmse
        top_param_names = list(sorted(list(param_name_to_values.keys()), key=key))[:self.nb_top_hyperparameters]
        top_emulator_with_search_marginal: EmulatorWithSearch = param_name_to_emulator_with_search[top_param_names[0]]
        rmse_validation_from_marginal_search = top_emulator_with_search_marginal.selected_validation_rmse
        log_info(f'Top Params: {top_param_names}')
        log_info(f'Top RMSE validation: {[key(param_name) for param_name in top_param_names]}')
        # Run a random search with respect to these top hyperparameters
        log_info(f'Start random search with: {top_param_names}')
        param_grid = {param_name: param_value for param_name, param_value in param_name_to_values.items()
                      if param_name in top_param_names}
        params_search = {'param_grid': param_grid, 'search_style': 'random', 'n_jobs': -1, 'n_iter': self.n_iter}
        emulator_with_search_random = EmulatorWithSearch(**self.params_emulator, **params_search)
        fit(emulator_with_search_random, self.dataset)
        rmse_validation_from_random_search = emulator_with_search_random.selected_validation_rmse
        log_info(f'Top RMSE validation from random search={rmse_validation_from_random_search}')
        # Return the emulator that performs best on the validation set
        if  rmse_validation_from_random_search < rmse_validation_from_marginal_search:
            log_info(f'Emulator from random search performs best')
            return emulator_with_search_random
        else:
            log_info(f'Emulator from marginal search performs best')
            return top_emulator_with_search_marginal

    @cached_property
    def params_emulator(self) -> dict[str, Any]:
        params_emulator: dict[str, Any] = {'model_selection': 'validated'}
        if self.fast:
            params_emulator['niterations'] = 2
        return params_emulator

    @property
    def fast(self) -> bool:
        """If nb_hyperparameters is not None, it means we want to run fast"""
        return self.nb_hyperparameters is not None

    def get_param_name_to_values(self) -> dict[str, list]:
        param_name_to_values = get_param_name_to_values()
        if self.nb_hyperparameters is not None:
            assert isinstance(self.nb_hyperparameters, int)
            first_param_names = list(sorted(list(param_name_to_values.keys())))[:self.nb_hyperparameters]
            param_name_to_values = {param_name: values for param_name, values in param_name_to_values.items()
                    if param_name in first_param_names}
        assert len(param_name_to_values) >= self.nb_top_hyperparameters
        if self.fast:
            param_name_to_values = {param_name: values[4:6] for param_name, values in param_name_to_values.items()}
        return param_name_to_values

    @property
    def series_summary(self) -> pd.Series:
        d = OrderedDict()
        d['RMSE test'] = self.rmse_test
        d['complexity'] = self.emulator.selected_complexity
        d['Equation'] = self.emulator.selected_expr
        s = pd.Series(data=d)
        s.name = self.validation_name
        return s

    @property
    def validation_name(self) -> str:
        return f'{self.validation_split} {self.validation_size}'




