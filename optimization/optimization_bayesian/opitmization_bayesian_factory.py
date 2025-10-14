from dataclasses import dataclass
from typing import Optional, Any, OrderedDict

import numpy as np
import optuna
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization
from utils.utils_log import log_info


def optimization_bayesian_factory(n_iter: int, nb_top_hyperparameters: Optional[int] =  None):
    @dataclass
    class OptimizationBayesian(Optimization):

        def __post_init__(self):
            super().__post_init__()
            self.n_iter = n_iter
            self.nb_top_hyperparameters = nb_top_hyperparameters
            # Modify self.param_name_to_values
            if self.nb_top_hyperparameters is not None:
                assert isinstance(self.nb_top_hyperparameters, int)
                assert isinstance(self.param_name_to_values, OrderedDict)
                top_param_names = list(self.param_name_to_values.keys())[:self.nb_top_hyperparameters]
                self.param_name_to_values = {param_name: param_values
                                             for param_name, param_values in self.param_name_to_values.items()
                                             if param_name in top_param_names}

        def get_budget(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                       variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                       y_units: Optional[ArrayLike[str]] = None) -> int:
            return self.n_iter


        def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> Emulator:
            log_info(f'Run {self.name}')

            def objective(trial):
                emulator = Emulator(**self.get_params(trial))
                emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
                return emulator.selected_validation_loss

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=self.n_iter, n_jobs=self.n_jobs)
            emulator = Emulator(**study.best_params)
            emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
            return emulator

        def get_params(self, trial) -> dict[str, Any]:
            params = {}
            for param_name, param_values in self.param_name_to_values.items():
                first_value = param_values[0]
                if (first_value is None) or (isinstance(first_value, str) or (isinstance(first_value, list))):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(first_value, int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                elif isinstance(first_value, float):
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    raise NotImplementedError(f'{param_values} for type {type(param_values[0])}')
            return params

        """ Properties for logs, plots"""

        @property
        def subclass_id(self) -> str:
            return f'bayes_{self.nb_top_hyperparameters}_{self.n_iter}'

        """ Properties for logs, plots"""


        @property
        def name(self):
            which_parameters = 'all' if self.nb_top_hyperparameters is None else f'top{self.nb_top_hyperparameters}'
            return f"bayesian search for {self.n_iter} samples, with {which_parameters} hyperparameters"

        @property
        def _label(self):
            return f"{self.n_iter} bayesian samples with\ntop{self.nb_top_hyperparameters} hyperparameters"

        @classmethod
        def color(cls):
            return 'dark'

        @classmethod
        def legend_label(cls):
            return "Bayesian search"

    OptimizationBayesian.__name__ += f'_{n_iter}_{nb_top_hyperparameters}'
    return OptimizationBayesian