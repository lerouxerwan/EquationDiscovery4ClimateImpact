from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization


def optimization_pipeline_factory(optimization_types: list[type]):

    class OptimizationPipeline(Optimization):


        def __post_init__(self):
            super().__post_init__()
            self.optimization_types = optimization_types

        def run(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> tuple[Emulator, dict[str, list] | None]:
            top_emulators = []
            param_name_to_values = self.param_name_to_values
            for optimization_type in self.optimization_types:
                optimization = optimization_type(self.model_selection, param_name_to_values)
                top_emulator, param_name_to_values  = optimization.run(X, y, validation_mask, variable_names, X_units, y_units)
                top_emulators.append(top_emulator)
            sorted_top_emulators = sorted(top_emulators, key=lambda emulator: emulator.selected_validation_loss)
            top_emulator = sorted_top_emulators[0]
            return top_emulator, param_name_to_values

        def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> Emulator:
            return self.run(X, y, validation_mask, variable_names, X_units, y_units)[0]



    return OptimizationPipeline


