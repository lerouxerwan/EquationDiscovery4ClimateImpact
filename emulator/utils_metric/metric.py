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
    MRE_AVERAGE_FIRST_20_YEARS = 5
    MRE_AVERAGE_LAST_20_YEARS = 6
    MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS = 7

metric_to_function = {
    Metric.MSE: mean_squared_error,
    Metric.MAE: mean_absolute_error,
    Metric.MRAE: mean_relative_absolute_error,
    Metric.COR: correlation,
    Metric.RMSE: root_mean_squared_error,
    Metric.MRE_AVERAGE_FIRST_20_YEARS: mean_relative_error_first_20_years,
    Metric.MRE_AVERAGE_LAST_20_YEARS: mean_relative_error_last_20_years,
    Metric.MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS: mean_relative_error_trend_between_first_and_last_20_years,
}


metric_to_label = {
    Metric.MSE: 'Mean squared error',
    Metric.MAE: 'Mean absolute error',
    Metric.MRAE: 'Mean relative absolute error',
    Metric.COR: 'Correlation',
    Metric.RMSE: 'Root mean squared error',
    Metric.MRE_AVERAGE_FIRST_20_YEARS: 'Mean relative error of the average on the first 20 years',
    Metric.MRE_AVERAGE_LAST_20_YEARS: 'Mean relative error of the average on the last 20 years',
    Metric.MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS: 'Mean relative error of the trend between first and last 20 years',
}

metric_to_str = {
    Metric.MSE: 'MSE',
    Metric.MAE: 'MAE',
    Metric.MRAE: 'MRAE',
    Metric.COR: 'COR',
    Metric.RMSE: 'RMSE',
    Metric.MRE_AVERAGE_FIRST_20_YEARS: 'MRE first 20 years',
    Metric.MRE_AVERAGE_LAST_20_YEARS: 'MRE last 20 years',
    Metric.MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS: 'MRE trend 20 years',
}

str_to_metric = {v: k for k, v in metric_to_str.items()}

def is_metric_with_percentage(metric: Metric) -> bool:
    return metric in [Metric.MRAE, Metric.MRE_AVERAGE_FIRST_20_YEARS, Metric.MRE_AVERAGE_LAST_20_YEARS,
                      Metric.MRE_TREND_BETWEEN_FIRST_AND_LAST_20_YEARS]

