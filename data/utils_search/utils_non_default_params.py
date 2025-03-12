from typing import Any

from sklearn.base import BaseEstimator


def get_non_default_params(estimator: BaseEstimator) -> dict[str, Any]:
    """Return a dictionary that maps each the name of each non default parameter to its non default value"""
    default_params = type(estimator)().get_params()
    non_default_params = {}
    for param_name, param_value in estimator.get_params().items():
        default_value = default_params[param_name]
        if param_value != default_value:
            non_default_params[param_name] = param_value
    return non_default_params