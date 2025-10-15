import warnings

import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import mean_squared_error


def mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    biases_percentage = [np.abs((pred - true) / true) * 100 if true != 0 else 0 for true, pred in
                         zip(y_true, y_pred)]
    return np.sum(biases_percentage) / len(biases_percentage)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median([np.abs(true - pred) for true, pred in zip(y_true, y_pred)]))


def spread_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.std(y_pred) / np.std(y_true)

def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert (1 <= y_true.ndim <= 2)
    if y_true.ndim == 2:
        y_true = y_true[:, 0]
    assert (1 <= y_pred.ndim <= 2)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    assert len(y_true) == len(y_pred)
    warnings.filterwarnings("error")
    try:
        res = pearsonr(y_true, y_pred)[0]
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

def mean_relative_error_first_20_years(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_first, _ = compute_climatological_averages(y_true)
    y_pred_first, _ = compute_climatological_averages(y_pred)
    return mean_relative_error(y_true_first, y_pred_first)

def mean_relative_error_last_20_years(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, y_true_last = compute_climatological_averages(y_true)
    _, y_pred_last = compute_climatological_averages(y_pred)
    return mean_relative_error(y_true_last, y_pred_last)

def mean_relative_error_trend_between_first_and_last_20_years(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_first, y_true_last = compute_climatological_averages(y_true)
    y_pred_first, y_pred_last = compute_climatological_averages(y_pred)
    return mean_relative_error(y_true_last - y_true_first, y_pred_last - y_pred_first)




