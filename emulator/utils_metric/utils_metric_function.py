import warnings

import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import mean_squared_error


def mean_relative_absolute_error(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    biases_percentage = [np.abs((pred - true) / true) * 100 if true != 0 else 0 for true, pred in
                         zip(y_true, y_predict)]
    return np.sum(biases_percentage) / len(biases_percentage)


def root_mean_squared_error(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_predict))

def correlation(u: np.ndarray, v: np.ndarray) -> float:
    assert (1 <= u.ndim <= 2)
    if u.ndim == 2:
        u = u[:, 0]
    assert (1 <= v.ndim <= 2)
    if v.ndim == 2:
        v = v[:, 0]
    assert len(u) == len(v)
    warnings.filterwarnings("error")
    try:
        res = pearsonr(u, v)[0]
    except ConstantInputWarning:
        res = -1
    warnings.resetwarnings()
    return res


