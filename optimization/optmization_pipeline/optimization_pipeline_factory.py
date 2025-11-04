from typing import Optional, OrderedDict

from numpy import ndarray
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.optimization import Optimization


def optimization_pipeline_factory(optimization_types: list[type]):

    class OptimizationPipeline(Optimization):

        def __post_init__(self):
            super().__post_init__()
            self.optimization_types = optimization_types

        def run(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> tuple[Emulator, dict[str, list] | None]:
            top_emulators = []
            param_name_to_values = self.param_name_to_values
            for optimization_type in self.optimization_types:
                optimization = optimization_type(self.model_selection, param_name_to_values, self.n_jobs,
                                                 self.timeout_in_seconds, self.gaussian_fit, self.interpretable_mode)
                top_emulator, param_name_to_values  = optimization.run(X, y, validation_mask, variable_names, X_units, y_units)
                top_emulators.append(top_emulator)
            sorted_top_emulators = sorted(top_emulators, key=lambda emulator: emulator.selected_loss_validation)
            top_emulator = sorted_top_emulators[0]
            return top_emulator, param_name_to_values

        def get_top_emulator(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                             y_units: Optional[ArrayLike[str]] = None) -> Emulator:
            return self.run(X, y, validation_mask, variable_names, X_units, y_units)[0]

        @property
        def optimizations_fake(self) -> list[Optimization]:
            # In order to get the budget without running the whole pipeline,
            # we instantiate each optimization_type with the initial self.param_name_to_values
            param_name_to_value_fake = OrderedDict()
            for param_name, param_values in self.param_name_to_values.items():
                param_name_to_value_fake[param_name] = param_values
            return [optimization_type(self.model_selection, param_name_to_value_fake)
                    for optimization_type in self.optimization_types]

        def get_budget(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                       variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                       y_units: Optional[ArrayLike[str]] = None) -> int:
            budgets = [optimization.get_budget(X, y, validation_mask, variable_names, X_units, y_units)
                       for optimization in self.optimizations_fake]
            print(budgets)
            return sum(budgets)

        @property
        def subclass_id(self) -> str:
            return '_'.join([opt.opt_id for opt in self.optimizations_fake])

        @property
        def name(self):
            return self.subclass_id

        @property
        def _label(self):
            return self.subclass_id

    return OptimizationPipeline


