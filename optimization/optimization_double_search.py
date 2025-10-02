from dataclasses import dataclass

import numpy as np
from pysr.utils import ArrayLike
from typing_extensions import Optional

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_optimization import get_loss
from utils.utils_log import log_info


@dataclass
class OptimizationDoubleSearch(Optimization):
    nb_top_hyperparameters: int =  5
    n_iter: int = 100

    @property
    def name(self):
        return f"random search with top{self.nb_top_hyperparameters} hyperparameters from the marginal search"

    @property
    def subclass_id(self) -> str:
        return f'{self.nb_top_hyperparameters}_{self.n_iter}'

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run {self.name}')
        top_param_names = self.get_top_param_names(X, y, validation_mask, variable_names, X_units, y_units)
        log_info(f'Top parameters are: {top_param_names}')
        param_grid = {param_name: param_values for param_name, param_values in self.param_name_to_values.items()
                      if param_name in top_param_names}
        params_emulator = {'model_selection': self.model_selection}
        params_search = {'param_grid': param_grid, 'search_style': 'random', 'n_iter': self.n_iter, 'n_jobs': None}
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator

    def get_top_param_names(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> list[str]:
        # Map each param_name to its validation rmse
        param_name_to_validation_rmse = dict()
        param_names = sorted(self.param_name_to_values.keys())
        for param_name in param_names:
            optimization = OptimizationMarginalSearch(self.model_selection, self.params_ranges, param_name)
            validation_loss = get_loss(optimization, X, y, validation_mask, variable_names, X_units, y_units)
            param_name_to_validation_rmse[param_name] = validation_loss
        # Compute the list of top param names
        sorted_param_names = list(sorted(param_names, key=lambda param_name: param_name_to_validation_rmse[param_name]))
        top_param_names = sorted_param_names[:self.nb_top_hyperparameters]
        log_info('Ranking from marginal search:')
        for rank, param_name in enumerate(sorted_param_names, 1):
            log_info(f'#{rank}: {param_name} with RMSE validation = {param_name_to_validation_rmse[param_name]}')
        return top_param_names
