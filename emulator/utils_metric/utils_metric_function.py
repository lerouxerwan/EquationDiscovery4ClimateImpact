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

def mean_relative_error(value_true: float, predict_value: float) -> float:
    assert isinstance(value_true, float) and isinstance(predict_value, float)
    return 100 * (predict_value - value_true) / value_true if value_true != 0 else 0

def condition_for_climatological_metrics(y_true: np.ndarray) -> bool:
    """To compute climatological metrics, at least 40 years of data are needed"""
    return len(y_true) >= 40

def compute_climatological_averages(y: np.ndarray) -> tuple[float, float]:
    assert condition_for_climatological_metrics(y)
    return float(np.mean(y[:20])), float(np.mean(y[-20:]))

def mean_relative_error_first_20_years(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    y_true_first, _ = compute_climatological_averages(y_true)
    y_predict_first, _ = compute_climatological_averages(y_predict)
    return mean_relative_error(y_true_first, y_predict_first)

def mean_relative_error_last_20_years(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    _, y_true_last = compute_climatological_averages(y_true)
    _, y_predict_last = compute_climatological_averages(y_predict)
    return mean_relative_error(y_true_last, y_predict_last)

def mean_relative_error_trend_between_first_and_last_20_years(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    y_true_first, y_true_last = compute_climatological_averages(y_true)
    y_predict_first, y_predict_last = compute_climatological_averages(y_predict)
    return mean_relative_error(y_true_last - y_true_first, y_predict_last - y_predict_first)




