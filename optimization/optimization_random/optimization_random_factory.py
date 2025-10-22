from dataclasses import dataclass

from numpy import ndarray
from pysr.utils import ArrayLike
from typing_extensions import Optional, OrderedDict

from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from utils.utils_log import log_info


def optimization_random_factory(n_iter: int, nb_top_hyperparameters: Optional[int] =  None):
    @dataclass
    class OptimizationRandom(Optimization):

        def __post_init__(self):
            super().__post_init__()
            self.n_iter = n_iter
            self.nb_top_hyperparameters = nb_top_hyperparameters

        def get_top_emulator(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> Emulator:
            log_info(f'Run {self.name}')
            params_emulator = {'model_selection': self.model_selection, 'timeout_in_seconds': self.timeout_in_seconds}
            params_search = {'param_grid': self.param_grid, 'search_style': 'random', 'n_iter': self.n_iter, 'n_jobs': self.n_jobs}
            emulator = EmulatorWithSearch(**params_emulator, **params_search)
            emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
            return emulator

        def get_budget(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                       variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                       y_units: Optional[ArrayLike[str]] = None) -> int:
            return self.n_iter

        @property
        def param_grid(self):
            if self.nb_top_hyperparameters is None:
                return self.param_name_to_values
            else:
                assert isinstance(self.nb_top_hyperparameters, int)
                assert isinstance(self.param_name_to_values, OrderedDict)
                top_param_names = list(self.param_name_to_values.keys())[:self.nb_top_hyperparameters]
                return {param_name: param_values for param_name, param_values in self.param_name_to_values.items()
                         if param_name in top_param_names}

        @property
        def subclass_id(self) -> str:
            return f'{self.nb_top_hyperparameters}_{self.n_iter}'

        """ Properties for logs, plots"""


        @property
        def name(self):
            which_parameters = 'all' if self.nb_top_hyperparameters is None else f'top{self.nb_top_hyperparameters}'
            return f"random search for {self.n_iter} with {which_parameters} hyperparameters"

        @property
        def _label(self):
            return f"{self.n_iter} random samples with\ntop{self.nb_top_hyperparameters} hyperparameters"

        @classmethod
        def color(cls):
            return 'darkgreen'

        @classmethod
        def legend_label(cls):
            return "Random search"


    OptimizationRandom.__name__ += f'_{n_iter}_{nb_top_hyperparameters}'
    return OptimizationRandom