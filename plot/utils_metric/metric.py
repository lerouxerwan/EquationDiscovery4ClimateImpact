from enum import Enum

import numpy as np
from numpy import ndarray
from sklearn.metrics import mean_squared_error, mean_absolute_error

from plot.utils_metric.utils_metric_function import correlation, \
    root_mean_squared_error, mean_relative_absolute_error, \
    median_absolute_error, spread_ratio


class Metric(Enum):
    MSE = 0
    MAE = 1
    MRAE = 2
    COR = 3
    RMSE = 4
    MEDAE = 5
    SPREADRATIO = 6
    NLL = 7

def compute_loss(y_true: ndarray, y_predicted: ndarray, metric: Metric) -> float:
    """Compute loss for a given metric, if the computation raises a ValueError we return np.nan as result"""
    loss_function = metric_to_function[metric]
    try:
        return loss_function(y_true=y_true, y_pred=y_predicted)
    except ValueError:
        return  np.nan

metric_to_function = {
    Metric.MSE: mean_squared_error,
    Metric.MAE: mean_absolute_error,
    Metric.MRAE: mean_relative_absolute_error,
    Metric.COR: correlation,
    Metric.RMSE: root_mean_squared_error,
    Metric.MEDAE: median_absolute_error,
    Metric.SPREADRATIO: spread_ratio,
}


metric_to_label = {
    Metric.MSE: 'Mean squared error',
    Metric.MAE: 'Mean absolute error',
    Metric.MRAE: 'Mean relative absolute error',
    Metric.COR: 'Correlation',
    Metric.RMSE: 'Root mean squared error',
    Metric.MEDAE: 'Median absolute error',
    Metric.SPREADRATIO: 'Spread ratio',
    Metric.NLL: 'Negative log likelihood',
}

metric_to_str = {
    Metric.MSE: 'MSE',
    Metric.MAE: 'MAE',
    Metric.MRAE: 'MRAE',
    Metric.COR: 'COR',
    Metric.RMSE: 'RMSE',
    Metric.MEDAE: 'MEDAE',
    Metric.SPREADRATIO: 'SR',
    Metric.NLL: 'NLL',
}

str_to_metric = {v: k for k, v in metric_to_str.items()}

def is_metric_with_percentage(metric: Metric) -> bool:
    return metric in [Metric.MRAE]

def is_metric_with_target_unit(metric: Metric) -> bool:
    return metric in {Metric.RMSE, Metric.MAE, Metric.MEDAE}


