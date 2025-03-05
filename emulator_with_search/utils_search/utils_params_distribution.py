from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import uniform


def get_param_distributions(param_grid: dict[str, Any]) -> dict[str, Any]:
    """A uniform distribution between min and max value is created for each param with more than 1 value"""
    param_distributions = {}
    for param_name, param_values in param_grid.items():
        if len(param_values) == 1:
            param_distributions[param_name] = param_values
        else:
            param_distributions[param_name] = get_param_distribution(param_values)
    return param_distributions


def get_param_distribution(param_values: list[int | float]):
    assert isinstance(param_values[0], (int, float))
    cast_as_int = isinstance(param_values[0], int)
    return ScaledUniformDistribution(min_value=min(param_values), max_value=max(param_values), cast_as_int=cast_as_int)


@dataclass
class ScaledUniformDistribution(object):
    min_value: float
    max_value: float
    cast_as_int: bool

    def rvs(self, size: int = 1, random_state=None) -> float | list[float]:
        scaled_min_value, scaled_max_value = np.log(self.min_value), np.log(self.max_value)
        # Using the parameters loc and scale, one obtains the uniform distribution on [loc, loc + scale].
        uniform_distribution = uniform(loc=scaled_min_value, scale=scaled_max_value - scaled_min_value)
        # Sample from this distribution
        scaled_values = uniform_distribution.rvs(size=size, random_state=random_state)
        sampled_values = [np.exp(scaled_value) for scaled_value in scaled_values]
        if self.cast_as_int:
            sampled_values = [int(sampled_value) for sampled_value in sampled_values]
        return sampled_values[0] if size == 1 else sampled_values
