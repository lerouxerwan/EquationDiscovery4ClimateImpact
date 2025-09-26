from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from emulator.emulator import Emulator


class Optimization(ABC):

    @abstractmethod
    def get_top_emulator(self, X: np.ndarray, y: np.ndarray, validation_mask: np.ndarray[bool],
                          variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                          y_units: Optional[ArrayLike[str]] = None) -> Emulator:
        pass

    @property
    @abstractmethod
    def name(self):
        pass
