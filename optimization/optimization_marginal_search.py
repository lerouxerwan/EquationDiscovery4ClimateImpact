from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from projects.paper.section_results.utils_hyperparameters import get_param_name_to_values
from utils.utils_log import log_info


@dataclass
class OptimizationMarginalSearch(Optimization):
    model_selection: Literal["best", "accuracy", "score", "validated"]
    param_name: str

    def __post_init__(self):
        param_name_to_values = get_param_name_to_values()
        assert self.param_name in param_name_to_values
        self.param_values = param_name_to_values[self.param_name]

    @property
    def name(self):
        return f"Marginal search for the hyperparameter '{self.param_name}'"

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run marginal search for {self.param_name}')
        params_emulator = {'model_selection': self.model_selection}
        params_search = {'param_grid': {self.param_name: self.param_values}, 'search_style': 'grid', 'n_jobs': None}
        emulator = EmulatorWithSearch(**params_emulator, **params_search)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator



