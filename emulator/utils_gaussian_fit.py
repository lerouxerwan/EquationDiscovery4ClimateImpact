from enum import StrEnum
from functools import partial
from math import exp
from typing import Callable

import numpy as np
from math import log
from numpy import ndarray

class UncertaintyInterval(StrEnum):
    plus_and_minus_std = '+-std'
    ninety_percent = '90%'

def get_loss_str_gaussian_fit(y_variable_name: str):
    # return f'log(sigma_value) + (({y_variable_name} - mu_value) / sigma_value) + exp(- (({y_variable_name} - mu_value) / sigma_value)) '
    return f'log(sigma_value) + ({y_variable_name} - mu_value)^2 / (2 * sigma_value^2)'

def _compute_loss_gaussian_fit(y_value: float, mu_value: float, sigma_value: float):
    # return log(sigma_value) + ((y_value - mu_value) / sigma_value) + exp(-((y_value - mu_value) / sigma_value))
    return log(sigma_value) + (y_value - mu_value)**2 / (2 * sigma_value ** 2)

def compute_loss_gaussian_fit(y: ndarray, mu: ndarray, sigma: ndarray):
    return np.mean([_compute_loss_gaussian_fit(*triple) for triple in zip(y, mu, sigma)])

def get_X_for_gaussian_fit(X: ndarray, y: ndarray) -> ndarray:
    return np.concat([X, np.expand_dims(y, axis=1)], axis=1)

def get_lambda_function_kwargs(s: str) -> Callable:
    return lambda **kwargs: eval(s, {}, kwargs)

def get_lambda_function_list(s: str, variable_names: list[str], add_exponential: bool = False) -> Callable:
    return partial(lambda_function, s=s, variable_names=variable_names, add_exponential=add_exponential)


def lambda_function(x: list, s: str, variable_names: list[str], add_exponential: bool = False):
    if add_exponential:
        return exp(eval(s, {}, dict(zip(variable_names, x))))
    else:
        return eval(s, {}, dict(zip(variable_names, x)))

