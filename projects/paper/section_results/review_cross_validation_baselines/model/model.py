from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class Model(ABC):
    """Fit a model without any preprocessing"""

    @property
    def estimator_type(self) -> type:
        raise NotImplementedError

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        raise NotImplementedError



