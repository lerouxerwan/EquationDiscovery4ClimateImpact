from abc import ABC
from dataclasses import dataclass
from typing import Any

from utils.utils_run import random_seed


@dataclass
class Model(ABC):
    """Fit a model without any preprocessing"""

    def estimator(self):
        return self.estimator_type(random_state=random_seed)

    @property
    def estimator_type(self) -> type:
        raise NotImplementedError

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        raise NotImplementedError

    @property
    def name(self):
        return self.estimator_type.__name__



