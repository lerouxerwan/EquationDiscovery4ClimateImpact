from dataclasses import dataclass
from typing import Literal

import numpy as np
from pysr.utils import ArrayLike
from typing_extensions import Optional

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from projects.paper.section_results.utils_hyperparameters import get_param_name_to_values
from slurm.nested_cv.utils_param_names import param_names
from utils.utils_log import log_info


@dataclass
class OptimizationDoubleSearch(Optimization):
    model_selection: Literal["best", "accuracy", "score", "validated"]
    nb_top_hyperparameters: str
    n_iter: int = 1000

    @property
    def name(self):
        return f"Random search with top{self.nb_top_hyperparameters} hyperparameters from the marginal search"

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        top_param_names = self.get_top_param_names(X, y, validation_mask, variable_names, X_units, y_units)
        log_info(f'Run random search with: {top_param_names}')
        param_grid = {param_name: param_values for param_name, param_values in get_param_name_to_values().items()
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
        for param_name in param_names:
            marginal_optimization = OptimizationMarginalSearch(self.model_selection, param_name)
            emulator = marginal_optimization.get_top_emulator(X, y, validation_mask, variable_names, X_units, y_units)
            param_name_to_validation_rmse[param_name] = emulator.selected_validation_rmse
        # Compute the list of top param names
        sorted_param_names = list(sorted(param_names, key=lambda param_name: param_name_to_validation_rmse[param_name]))
        top_param_names = sorted_param_names[:self.nb_top_hyperparameters]
        log_info('Ranking from marginal search:')
        for rank, param_name in enumerate(sorted_param_names, 1):
            log_info(f'#{rank}: {param_name} with RMSE validation = {param_name_to_validation_rmse[param_name]}')
        return top_param_names
