from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import optuna
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization
from utils.utils_log import log_info


@dataclass
class OptimizationOptuna(Optimization):
    model_selection: Literal["best", "accuracy", "score", "validated"]
    n_trials: int = 50

    @property
    def name(self):
        return f"optuna search with {self.n_trials} trials"

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                         variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                         y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        log_info(f'Run {self.name}')
        def objective(trial):
            emulator = Emulator(
                niterations=trial.suggest_int("niterations", 10, 20),
                maxsize=trial.suggest_int("maxsize", 20, 30),
            )
            emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
            return emulator.selected_validation_loss
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.n_trials)
        return Emulator(**study.best_params).fit(X, y, validation_mask, variable_names, X_units, y_units)
