from typing import Optional

import numpy as np
import pandas as pd

from emulator.pysr_emulator import PySREmulator
from emulator.utils_metric.metric import Metric, metric_to_function, metric_to_label
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from utils.utils_plot import compute_axis_lim


def plot_climato(emulator: PySREmulator, X_train: np.ndarray | pd.DataFrame,
                 y_train: np.ndarray | pd.Series, X_test: Optional[np.ndarray | pd.DataFrame] = None,
                 y_test: Optional[np.ndarray | pd.Series] = None,
                 years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                 rcp_name_test: Optional[str]=None, ind_validation:Optional[np.ndarray[bool]] = None,
                 target_label: str = "Target (-)", show: bool = False):
    y_train_predicted = emulator.predict(X_train)
    y_test_predicted = None if X_test is None else emulator.predict(X_test)
    y_values = np.concat([y for y in [y_train, y_test, y_train_predicted, y_test_predicted] if y is not None])
    ymin_and_ymax = compute_axis_lim(y_values)
    # Plot observation and prediction using the same limit
    rcp_name_to_observed_std_values_and_years_and_color = _plot_climato(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test,
                  ind_validation, target_label, "Observed", show, ymin_and_ymax)
    rcp_name_to_predicted_std_values_and_years_and_color = _plot_climato(y_train_predicted, y_test_predicted, years_train, years_test, rcp_name_train, rcp_name_test,
                  ind_validation, target_label, "Predicted", show, ymin_and_ymax)
    # Plot ratio of std
    rcp_name_to_list_of_years_and_y_and_color_and_label = {}
    for rcp_name, (observed_std_values, years, color) in rcp_name_to_observed_std_values_and_years_and_color.items():
        predicted_std_values, years, color = rcp_name_to_predicted_std_values_and_years_and_color[rcp_name]
        values = [predicted_v / observed_v for observed_v, predicted_v in zip(observed_std_values, predicted_std_values)]
        prefix = 'Std 30-years predictions divided by std 30-years observations'
        rcp_name_to_list_of_years_and_y_and_color_and_label[rcp_name] = [(years, values, color, prefix)]
    plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, f"{prefix}\nfor",
                                    show)



def _plot_climato(y_train: np.ndarray | pd.Series, y_test: Optional[np.ndarray | pd.Series] = None,
                  years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                  rcp_name_test: Optional[str]=None, ind_validation:Optional[np.ndarray[bool]] = None,
                  target_label: str = "Target (-)", prefix: str = "", show: bool = False, ymin_and_ymax: Optional[tuple[float, float]] = None):
    """Plot several RCP climatological time series on the same graph"""
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, ind_validation)
    return plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, prefix, show, ymin_and_ymax)


def plot_errors_climato(emulator: PySREmulator, X_train: np.ndarray | pd.DataFrame,
                        y_train: np.ndarray | pd.Series, X_test: Optional[np.ndarray | pd.DataFrame] = None,
                        y_test: Optional[np.ndarray | pd.Series] = None,
                        years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                        rcp_name_test: Optional[str]=None, ind_validation:Optional[np.ndarray[bool]] = None,
                        target_label: str = "Target (-)", show: bool = False):
    errors_train = compute_differences(emulator, X_train, y_train)
    errors_test = compute_differences(emulator, X_test, y_test)
    rcp_name_to_list_of_years_and_errors_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(errors_train, errors_test, years_train, years_test, rcp_name_train, rcp_name_test, ind_validation)
    plot_climatological_time_series(rcp_name_to_list_of_years_and_errors_and_color_and_label, errors_train, target_label,
                                    f'Error for', show)

def compute_differences(emulator: PySREmulator, X: Optional[np.ndarray | pd.DataFrame], y: Optional[np.ndarray | pd.Series]) -> Optional[np.ndarray]:
    if X is None:
        return None
    else:
        y_predicted = emulator.predict(X)
        errors = [y_predicted_value - y_value for y_value, y_predicted_value in zip(y, y_predicted)]
        return np.array(errors)






