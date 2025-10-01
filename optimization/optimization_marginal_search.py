from dataclasses import dataclass
from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from utils.utils_log import log_info


@dataclass
class OptimizationMarginalSearch(Optimization):
    param_name: str = 'niterations'

    @property
    def name(self):
        return f"marginal search for the hyperparameter '{self.param_name}'"

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run {self.name}')
        params_emulator = {'model_selection': self.model_selection}
        param_values = self.param_name_to_values[self.param_name]
        params_search = {'param_grid': {self.param_name: param_values}, 'search_style': 'grid', 'n_jobs': None}
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator



