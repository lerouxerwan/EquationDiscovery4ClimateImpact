import os.path as op
from typing import Optional, Any

from numpy import ndarray

from data.utils_run.utils_key import get_hash_str, get_hash_params
from utils.utils_path import RUN_PATH

CSV_FILENAME = 'cv_results.csv'
JSON_FILENAME = 'params_emulator.json'

def get_output_directory(X: ndarray, y: ndarray, validation_mask: Optional[ndarray]):
    """Directory, containing subdirectories with results, for a dataset and a validation size"""
    if validation_mask is None:
        dataset_folder = get_hash_str(X, y)
    else:
        dataset_folder = get_hash_str(X, y, validation_mask)
    return op.join(RUN_PATH, dataset_folder)

def get_run_id(params: dict) -> str:
    """Folder, whose name characterize the run using some hyperparameters"""
    return get_hash_str(get_hash_params(params))


def get_non_default_params(params: dict[str, Any], default_type: type) -> dict[str, Any]:
    """Return a dictionary that maps each the name of each non default parameter to its non default value"""
    default_params = default_type().get_params()
    non_default_params = {}
    for param_name, param_value in params.items():
        default_value = default_params[param_name]
        if param_value != default_value:
            non_default_params[param_name] = param_value
    return non_default_params
