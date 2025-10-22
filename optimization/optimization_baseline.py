from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization


@dataclass
class OptimizationBaseline(Optimization):
    """Optimization with Default Hyperparameters except Validated Model Selection"""

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray,
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        emulator = Emulator(model_selection=self.model_selection)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator
    
    @property
    def subclass_id(self) -> str:
        return "baseline"

    """ Properties for logs, plots"""

    def get_budget(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray,
                   variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                   y_units: Optional[ArrayLike[str]] = None) -> int:
        return 1

    @classmethod
    def color(cls):
        return "lightgreen"

    @classmethod
    def legend_label(cls):
        return "Baseline"

    @property
    def _label(self):
        return "Default hyperparameters"

    @property
    def name(self):
        return f"Baseline with model selection '{self.model_selection}'"
