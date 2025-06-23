from typing import Any


def get_non_default_params(current_params: dict[str, Any], default_type: type) -> dict[str, Any]:
    """Return a dictionary that maps each the name of each non default parameter to its non default value"""
    default_params = default_type().get_params()
    non_default_params = {}
    for param_name, param_value in current_params.items():
        default_value = default_params[param_name]
        if param_value != default_value:
            non_default_params[param_name] = param_value
    return non_default_params