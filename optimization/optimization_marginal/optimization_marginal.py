

from dataclasses import dataclass
from typing import Optional, OrderedDict

from numpy import ndarray
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from utils.utils_log import log_info


@dataclass
class OptimizationMarginal(Optimization):

    def run(self, X: ndarray, y: ndarray, validation_mask: ndarray,
            variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
            y_units: Optional[ArrayLike[str]] = None) -> tuple[Emulator, dict[str, list] | None]:
        top_emulator = self.get_top_emulator(X, y, validation_mask, variable_names, X_units, y_units)
        # Extract ordered list of param_names (first in the list has the lowest validation RMSE)
        params_list = top_emulator.run_.df_cv_results['params'].values
        assert all([len(params) == 1 for params in params_list])
        param_names = [list(params.keys())[0] for params in params_list]
        # Returns for param_name_to_values an OrderedDict, where the top params is first
        param_name_to_values = OrderedDict()
        for param_name in param_names:
            if param_name not in param_name_to_values:
                param_name_to_values[param_name] = self.param_name_to_values[param_name]
        return top_emulator, param_name_to_values

    def get_top_emulator(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run {self.name}')
        params_emulator = self.get_params_emulator(variable_names)
        params_search = {'param_grid': self.param_grid, 'search_style': 'grid', 'n_jobs': self.n_jobs}
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator

    def get_budget(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                   variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                   y_units: Optional[ArrayLike[str]] = None) -> int:
        list_of_values = [list(param_name_to_values.values())[0]  for param_name_to_values in self.param_grid]
        return sum([len(values) for values in list_of_values])

    @property
    def param_grid(self):
        return [{param_name: param_values} for param_name, param_values in self.param_name_to_values.items()]


    @property
    def subclass_id(self) -> str:
        return 'marginal'

    """ Properties for logs, plots"""

    @property
    def name(self):
        return f"marginal search for all hyperparameters'"

    @property
    def _label(self):
        return f"marginal '"

    @classmethod
    def color(cls):
        return 'limegreen'

    @classmethod
    def legend_label(cls):
        return "Marginal grid search"










