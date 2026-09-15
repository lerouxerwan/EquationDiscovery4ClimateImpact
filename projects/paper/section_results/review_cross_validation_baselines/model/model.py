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
        name = self.estimator_type.__name__
        name = name.replace('Regressor', '')
        return self.CamelSplitMethod(name)

    @staticmethod
    def CamelSplitMethod(string_value):
        result = [string_value[0]]
        for char in string_value[1:]:
            if char.isupper():
                result.extend([' ', char])
            else:
                result.append(char)
        return ''.join(result)


