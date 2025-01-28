from typing import Optional

import numpy as np
import pandas as pd

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.metric import Metric, metric_to_function, metric_to_label
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label


def plot_predicted_climato(emulator: ClimateImpactEmulator, X_train: np.ndarray | pd.Series,
                           X_test: Optional[np.ndarray | pd.Series] = None,
                           years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                           rcp_name_test: Optional[str]=None, nb_historical_years: int = 0,
                           target_label: str = "Target (-)", show: bool = False):
    plot_observed_climato(emulator.predict(X_train), None if X_test is None else emulator.predict(X_test),
                          years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years, target_label, "Predicted", show)

def plot_observed_climato(y_train: np.ndarray | pd.Series, y_test: Optional[np.ndarray | pd.Series] = None,
                          years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                          rcp_name_test: Optional[str]=None, nb_historical_years: int = 0,
                          target_label: str = "Target (-)", suffix: str = 'Observed', show: bool = False):
    """Plot several RCP climatological time series on the same graph"""
    rcp_name_to_list_of_years_and_y_and_label_and_color = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years)
    plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_label_and_color, y_train, target_label, suffix, show)

def plot_errors_climato(emulator: ClimateImpactEmulator, X_train: np.ndarray | pd.DataFrame,
                        y_train: np.ndarray | pd.Series, X_test: Optional[np.ndarray | pd.DataFrame] = None,
                        y_test: Optional[np.ndarray | pd.Series] = None,
                        years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                        rcp_name_test: Optional[str]=None, nb_historical_years: int = 0,
                        target_label: str = "Target (-)", metric: Metric = Metric.RMSE, show: bool = False):
    errors_train = compute_errors(emulator, X_train, y_train, metric)
    errors_test = compute_errors(emulator, X_test, y_test, metric)
    rcp_name_to_list_of_years_and_errors_label_and_color = load_rcp_name_to_list_of_years_and_y_and_color_and_label(errors_train, errors_test, years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years)
    plot_climatological_time_series(rcp_name_to_list_of_years_and_errors_label_and_color, errors_train, target_label,
                                    f'{metric_to_label[metric]} of', show)

def compute_errors(emulator: ClimateImpactEmulator, X: Optional[np.ndarray | pd.DataFrame], y: Optional[np.ndarray | pd.Series], metric: Metric) -> Optional[np.ndarray]:
    if X is None:
        return None
    else:
        metric_function = metric_to_function[metric]
        y_predicted = emulator.predict(X)
        errors = [metric_function([y_value], [y_predicted_value]) for y_value, y_predicted_value in zip(y, y_predicted)]
        return np.array(errors)





