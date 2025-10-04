from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any, Literal

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator
from optimization.utils_params.utils_params_values import ParamsValues, get_param_name_to_values


@dataclass
class Optimization(ABC):
    model_selection: str = 'best'
    params_values: Optional[ParamsValues] = None

    @abstractmethod
    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        pass

    @property
    def opt_id(self) -> str:
        return f'{self.model_selection}_{self.params_values}_{self.subclass_id}'

    @property
    @abstractmethod
    def subclass_id(self) -> str:
        pass

    @cached_property
    def param_name_to_values(self) -> Optional[dict[str, list[Any]]]:
        return get_param_name_to_values(self.params_values)

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
