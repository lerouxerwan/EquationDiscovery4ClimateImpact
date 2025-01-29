from enum import Enum

from sklearn.metrics import mean_squared_error, mean_absolute_error

from emulator.utils_metric.utils_metric_function import mean_relative_error, correlation, \
    root_mean_squared_error, mean_relative_error_first_20_years, mean_relative_absolute_error, \
    mean_relative_error_last_20_years, mean_relative_error_trend_between_first_and_last_20_years


class Metric(Enum):
    MSE = 0
    MAE = 1
    MRAE = 2
    COR = 3
    RMSE = 4

metric_to_function = {
    Metric.MSE: mean_squared_error,
    Metric.MAE: mean_absolute_error,
    Metric.MRAE: mean_relative_absolute_error,
    Metric.COR: correlation,
    Metric.RMSE: root_mean_squared_error,
}


metric_to_label = {
    Metric.MSE: 'Mean squared error',
    Metric.MAE: 'Mean absolute error',
    Metric.MRAE: 'Mean relative absolute error',
    Metric.COR: 'Correlation',
    Metric.RMSE: 'Root mean squared error',
}

metric_to_str = {
    Metric.MSE: 'MSE',
    Metric.MAE: 'MAE',
    Metric.MRAE: 'MRAE',
    Metric.COR: 'COR',
    Metric.RMSE: 'RMSE',
}

str_to_metric = {v: k for k, v in metric_to_str.items()}

def is_metric_with_percentage(metric: Metric) -> bool:
    return metric in [Metric.MRAE]

