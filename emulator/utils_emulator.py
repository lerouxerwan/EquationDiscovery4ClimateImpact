from math import exp
from typing import Callable

import numpy as np
from math import log
from numpy import ndarray

params_that_do_not_impact_the_fit_results = {'logger_spec', 'output_directory', 'run_id',
                                             'parallelism', 'procs', 'cluster_manager',
                                             'deterministic', 'verbosity', 'update_verbosity', 'progress',
                                             'input_stream', 'temp_equation_file', 'tempdir', 'delete_tempfiles', 'extra_sympy_mappings',
                                             'extra_torch_mappings', 'extra_jax_mappings', 'update', 'n_jobs',

                                                # The following params impact the results, but they both directly
                                             # depend on 'gaussian_fit' params, so  we do not need to include
                                             'expression_spec', 'elementwise_loss', 'loss_function'}

### Some methods for gaussian fit ###

def get_loss_str_gaussian_fit(y_variable_name: str):
    return f'log(sigma) + ({y_variable_name} - mu)^2 / (2 * sigma^2)'

def _compute_loss_gaussian_fit(y: float, mu: float, sigma: float):
    return log(sigma) + (y - mu)**2 / (2 * sigma**2)

def compute_loss_gaussian_fit(y: ndarray, mu: ndarray, sigma: ndarray):
    return np.mean([_compute_loss_gaussian_fit(*triple) for triple in zip(y, mu, sigma)])

def get_X_for_gaussian_fit(X: ndarray, y: ndarray) -> ndarray:
    return np.concat([X, np.expand_dims(y, axis=1)], axis=1)

def get_lambda_function_kwargs(s: str) -> Callable:
    return lambda **kwargs: eval(s, {}, kwargs)

def get_lambda_function_list(s: str, variable_names: list[str], add_exp: bool = False) -> Callable:
    if add_exp:
        return lambda *args: exp(eval(s, {}, dict(zip(variable_names, args[0]))))
    else:
        return lambda *args: eval(s, {}, dict(zip(variable_names, args[0])))

