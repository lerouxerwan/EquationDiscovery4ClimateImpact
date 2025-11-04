from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal, Any

from numpy import ndarray
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values


@dataclass
class Optimization(ABC):
    """Optimization with
    -3 parameters for each Emulator fit (model_selection, timeout_in_seconds, gaussian_fit)
    -2 parameters for the hyperparameter search
    """
    model_selection: Literal["best", "accuracy", "score", "validated"] = 'best'
    param_name_to_values: Optional[ParamNameToValues | dict[str, list]] = None
    n_jobs: int = 1
    timeout_in_seconds: float | None = None
    gaussian_fit: bool = False
    interpretable_mode: bool = False


    def get_params_emulator(self, variable_names: Optional[list[str]] = None):
        params_emulator = {'model_selection': self.model_selection, 'timeout_in_seconds': self.timeout_in_seconds}
        if self.gaussian_fit:
            params_emulator['gaussian_fit'] = True
            params_emulator['X_variable_names_for_gaussian_fit'] = variable_names
            params_emulator['y_variable_name_for_gaussian_fit'] = 'Target'
        if self.interpretable_mode:
            params_emulator['interpretable_mode'] = self.interpretable_mode
        return params_emulator

    def __post_init__(self):
        if isinstance(self.param_name_to_values, ParamNameToValues):
            self.param_name_to_values = get_param_name_to_values(self.param_name_to_values)
        assert (self.param_name_to_values is None) or isinstance(self.param_name_to_values, dict)

    def run(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> tuple[Emulator, dict[str, list] | None]:
        top_emulator = self.get_top_emulator(X, y, validation_mask, variable_names, X_units, y_units)
        return top_emulator, self.param_name_to_values

    @abstractmethod
    def get_top_emulator(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        pass

    @abstractmethod
    def get_budget(self, X: ndarray, y: ndarray, validation_mask: ndarray,
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> int:
        pass


    @property
    def opt_id(self) -> str:
        opt_id = f'{self.model_selection}_{self.subclass_id}'
        if self.timeout_in_seconds is not None:
            opt_id += f'_{self.timeout_in_seconds}'
        if self.gaussian_fit:
            opt_id += '_g'
        if self.interpretable_mode:
            opt_id += '_im'
        return opt_id

    @property
    @abstractmethod
    def subclass_id(self) -> str:
        pass

    """ Properties for logs, plots"""

    @property
    @abstractmethod
    def name(self):
        pass

    @classmethod
    def color(cls):
        return 'k'

    @classmethod
    def legend_label(cls):
        return ""

    @property
    def label(self):
        model_selection_str = '' if self.model_selection == 'best' else " with 'validated' method"
        return self._label + model_selection_str

    @property
    @abstractmethod
    def _label(self):
        pass
