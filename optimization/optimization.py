from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any, Literal

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values


@dataclass
class Optimization(ABC):
    model_selection: str = 'best'
    param_name_to_values: Optional[ParamNameToValues | dict[str, list]] = None

    def __post_init__(self):
        if isinstance(self.param_name_to_values, ParamNameToValues):
            self.param_name_to_values = get_param_name_to_values(self.param_name_to_values)
        assert (self.param_name_to_values is None) or isinstance(self.param_name_to_values, dict)

    def run(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> tuple[Emulator, dict[str, list] | None]:
        top_emulator = self.get_top_emulator(X, y, validation_mask, variable_names, X_units, y_units)
        return top_emulator, self.param_name_to_values

    @abstractmethod
    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        pass


    @abstractmethod
    def get_budget(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> int:
        pass


    @property
    def opt_id(self) -> str:
        return f'{self.model_selection}_{self.param_name_to_values}_{self.subclass_id}'

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
