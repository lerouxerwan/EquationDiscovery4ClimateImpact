from dataclasses import dataclass
from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from utils.utils_log import log_info


@dataclass
class OptimizationCoupleRandom(Optimization):
    param_name_1: str = 'niterations'
    param_name_2: str = 'maxsize'
    n_iter: int = 100

    def __post_init__(self):
        assert self.param_name_1 != self.param_name_2

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run {self.name}')
        params_emulator = {'model_selection': self.model_selection}
        param_grid = {param_name: self.param_name_to_values[param_name]
                      for param_name in [self.param_name_1, self.param_name_2]}
        params_search = {'param_grid': param_grid, 'search_style': 'random', 'n_iter': self.n_iter, 'n_jobs': None}
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator

    @property
    def subclass_id(self) -> str:
        return self.param_name_1 + '_' + self.param_name_2

    """ Properties for logs, plots"""

    @property
    def name(self):
        return f"couple search for the hyperparameters '{self.param_name_1}' and '{self.param_name_2}'"

    @property
    def _label(self):
        return (f"{len(self.param_name_to_values[self.param_name_1]) * len(self.param_name_to_values[self.param_name_2])} fixed values "
                f"for the\nhyperparameters '{self.param_name_1}' and '{self.param_name_2}'")

    @classmethod
    def color(cls):
        return 'darkorange'

    @classmethod
    def legend_label(cls):
        return "Couple search"










