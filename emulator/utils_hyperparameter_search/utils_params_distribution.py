from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import uniform


def get_param_distributions(param_grid: dict[str, Any]) -> dict[str, Any]:
    """A uniform distribution between min and max value is created for each param with more than 1 value"""
    param_distributions = {}
    for param_name, param_values in param_grid.items():
        if len(param_values) == 1:
            distribution = param_values
        elif isinstance(param_values[0], (int, float)):
            min_value, max_value = min(param_values), max(param_values)
            assert min_value != max_value, f'{param_name} {param_values}'
            uniform_distribution_without_scaling = max_value <= 1.
            uniform_distribution_without_scaling |= (max_value / min_value) < 10
            if uniform_distribution_without_scaling:
                distribution_type = UniformDistribution
            else:
                distribution_type = ScaledUniformDistribution
            distribution = distribution_type(min_value, max_value, cast_as_int=isinstance(min_value, int))
        else:
            # In this case, the distribution will sample from the list of parameter values
            distribution = param_values
        param_distributions[param_name] = distribution
    return param_distributions



@dataclass
class UniformDistribution(object):
    min_value: float
    max_value: float
    cast_as_int: bool


    def rvs(self, size: int = 1, random_state=None) -> float | list[float]:
        uniform_distribution = uniform(loc=self.min_value, scale=self.max_value - self.min_value)
        # Sample from this distribution
        sampled_values = uniform_distribution.rvs(size=size, random_state=random_state)
        if self.cast_as_int:
            sampled_values = [int(sampled_value) for sampled_value in sampled_values]
        return sampled_values[0] if size == 1 else sampled_values


@dataclass
class ScaledUniformDistribution(UniformDistribution):

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
