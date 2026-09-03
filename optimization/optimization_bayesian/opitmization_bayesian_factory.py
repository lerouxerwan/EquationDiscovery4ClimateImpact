import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Any, OrderedDict

import optuna
from numpy import ndarray
from optuna.samplers import TPESampler
from pysr.utils import ArrayLike

from data.utils_run.run import Run
from data.utils_run.utils_run import get_output_directory, get_run_id, remove_parameters_not_json_serializable
from emulator.emulator import Emulator
from emulator.emulator_with_search import EmulatorWithSearch
from optimization.optimization import Optimization
from utils.utils_log import log_info
from utils.utils_run import random_seed
# import timeout_decorator


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

        def get_budget(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                       variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                       y_units: Optional[ArrayLike[str]] = None) -> int:
            return self.n_iter


        def get_top_emulator(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> Emulator:
            log_info(f'Run {self.name}')

            # @timeout_decorator.timeout(60, timeout_exception=optuna.TrialPruned, use_signals=True)
            def objective(trial):
                objective_emulator = Emulator(**self.get_params(trial, variable_names))
                try:
                    objective_emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
                    return objective_emulator.selected_loss_validation
                except Exception as e:
                    log_info(f"Exception catch : {e}")
                    return math.inf

            # Load json filepath
            params_search = {'param_grid': self.param_name_to_values, 'search_style': 'TPESampler',
                             'n_iter': self.n_iter, 'n_jobs': self.n_jobs}
            emulator = EmulatorWithSearch(**self.get_params_emulator(), **params_search)
            output_directory = get_output_directory(X, y, validation_mask)
            run_id = get_run_id(EmulatorWithSearch.get_non_default_params(emulator.get_params()))
            run = Run(output_directory, run_id)
            filepath_pkl = run.filepath_best_params_for_optuna

            # Load or save the best params
            if os.path.exists(filepath_pkl):
                # Load best params
                log_info("Load best params of TPESampler from pickle")
                with open(filepath_pkl, 'rb') as f:
                    best_params = pickle.load(f)
            else:
                # TPESampler is the default sampler used by optuna, when sampler argument is None. For further explanations:
                # https://medium.com/@becaye-balde/bayesian-sorcery-for-hyperparameter-optimization-using-optuna-1ee4517e89a
                # Note that TPESampler is indeed a Bayesian optimization method
                # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_seed))
                sampler = TPESampler(seed=random_seed)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                study.optimize(objective, n_trials=self.n_iter, n_jobs=self.n_jobs)

                # Save best params
                best_params = study.best_params
                log_info("Save best params of TPESampler to a pickle")
                with open(filepath_pkl, 'wb') as f:
                    pickle.dump(best_params, f)

            # Fit with best params
            emulator = Emulator(**self.get_params_emulator(), **best_params)
            emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
            return emulator

        def get_params(self, trial, variable_names) -> dict[str, Any]:
            params = self.get_params_emulator(variable_names)
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