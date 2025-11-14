import os.path as op
from operator import itemgetter
from typing import Optional, Any

from numpy import ndarray

from utils.utils_hash import get_hash_str
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


def get_hash_params(params: dict[str, Any]) -> list[tuple[Any] | Any]:
    """Summarize all parameters as list (but do not include params that do not impact the fit results)"""
    params_loop = {k: v for k, v in params.items() if k not in params_that_do_not_impact_the_fit_results}
    entire_hash_params = []
    for k, v in sorted(list(params_loop.items()), key=itemgetter(0)):
        if isinstance(v ,(float, int)):
            hash_params = (k, v)
        elif isinstance(v, (list, str, tuple)):
            hash_params = tuple([k]) + tuple(v)
        elif isinstance(v, dict):
            hash_params =  tuple([k])  + tuple(get_hash_params(v))
        else:
            raise ValueError(f'For the key {k}, type(v)={type(v)} with v={v}')
        entire_hash_params.append(hash_params)
    return entire_hash_params


params_that_do_not_impact_the_fit_results = {'logger_spec', 'output_directory', 'run_id',
                                             'parallelism', 'procs', 'cluster_manager',
                                             'deterministic', 'verbosity', 'update_verbosity', 'progress',
                                             'input_stream', 'temp_equation_file', 'tempdir', 'delete_tempfiles', 'extra_sympy_mappings',
                                             'extra_torch_mappings', 'extra_jax_mappings', 'update', 'n_jobs',

                                                # The following params impact the results, but they both directly
                                             # depend on 'gaussian_fit' params, so  we do not need to include
                                             'expression_spec', 'elementwise_loss', 'loss_function'}

def remove_parameters_not_json_serializable(params):
    """Remove parameters that are not JSON serializable"""
    for param_name in param_names_not_handled_by_json:
        if param_name in params:
            params.pop(param_name)
    return params
            
param_names_not_handled_by_json = ['expression_spec', 'extra_sympy_mappings']

