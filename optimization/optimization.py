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
    params_ranges: Optional[ParamsValues] = None

    @cached_property
    def param_name_to_values(self) -> Optional[dict[str, list[Any]]]:
        return get_param_name_to_values(self.params_ranges)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        pass

    @property
    def opt_id(self) -> str:
        return f'{self.model_selection}_{self.params_ranges}_{self.subclass_id}'

    @property
    @abstractmethod
    def subclass_id(self) -> str:
        pass



