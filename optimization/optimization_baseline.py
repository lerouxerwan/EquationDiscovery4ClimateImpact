from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization


@dataclass
class OptimizationBaseline(Optimization):
    """Optimization with Default Hyperparameters except Validated Model Selection"""

    @property
    def name(self):
        return f"Baseline with model selection '{self.model_selection}'"

    @property
    def subclass_id(self) -> str:
        return "baseline"

    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        emulator = Emulator(model_selection=self.model_selection)
        emulator.fit(X, y, validation_mask, variable_names, X_units, y_units)
        return emulator


optimization_baseline_with_best_model_selection = OptimizationBaseline('best')
optimization_baseline_with_validated_model_selection = OptimizationBaseline('validated')