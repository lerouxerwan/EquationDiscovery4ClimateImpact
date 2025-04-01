from typing import Optional

import numpy as np
import pandas as pd

from emulator.pysr_emulator import PySREmulator
from emulator.utils_metric.metric import Metric, metric_to_function, metric_to_label
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label, \
    get_true_and_predicted_prefix, uncapitalize
from utils.utils_plot import compute_axis_lim


def plot_climato(emulator: PySREmulator, X_train: np.ndarray,
                 y_train: np.ndarray,
                 validation_mask: np.ndarray[bool],
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                 rcp_name_test: Optional[str]=None,
                 target_label: str = "Target (-)", show: Optional[bool] = False):
    y_train_predicted = emulator.predict(X_train)
    y_test_predicted = None if X_test is None else emulator.predict(X_test)
    y_values = np.concat([y for y in [y_train, y_test, y_train_predicted, y_test_predicted] if y is not None])
    ymin_and_ymax = compute_axis_lim(y_values)
    # Plot observation and prediction using the same limit
    true_target_label, predicted_target_label = get_true_label_and_predicted_label(target_label)
    rcp_name_to_years_and_observed_std_values_years_and_color = _plot_climato(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test,
                  validation_mask, true_target_label, show, ymin_and_ymax)
    rcp_name_to_years_and_predicted_std_values_and_color = _plot_climato(y_train_predicted, y_test_predicted, years_train, years_test, rcp_name_train, rcp_name_test,
                  validation_mask, predicted_target_label, show, ymin_and_ymax)
    # Plot ratio of std
    rcp_name_to_list_of_years_and_y_and_color_and_label = {}
    for rcp_name, (years, observed_std_values, color) in rcp_name_to_years_and_observed_std_values_years_and_color.items():
        years, predicted_std_values, color = rcp_name_to_years_and_predicted_std_values_and_color[rcp_name]
        values = [predicted_v / observed_v for observed_v, predicted_v in zip(observed_std_values, predicted_std_values)]
        true_prefix, predicted_prefix = get_true_and_predicted_prefix()
        metric = 'std on 30-years'
        prefix = f'{true_prefix} {metric} / {predicted_prefix} {metric}'
        rcp_name_to_list_of_years_and_y_and_color_and_label[rcp_name] = [(years, values, color, prefix)]
    plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, f"{prefix}\nfor",
                                    show, plot_std=False, suffix_plot_name='std')



def _plot_climato(y_train: np.ndarray, y_test: Optional[np.ndarray] = None,
                  years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                  rcp_name_test: Optional[str]=None, validation_mask:Optional[np.ndarray[bool]] = None,
                  target_label: str = "Target (-)", show: bool = False, ymin_and_ymax: Optional[tuple[float, float]] = None):
    """Plot several RCP climatological time series on the same graph"""
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, validation_mask)
    return plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, "", show, ymin_and_ymax, suffix_plot_name=target_label.split()[0])


def plot_errors_climato(emulator: PySREmulator, X_train: np.ndarray,
                        y_train: np.ndarray,
                        validation_mask: np.ndarray[bool],
                        X_test: Optional[np.ndarray] = None,
                        y_test: Optional[np.ndarray] = None,
                        years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                        rcp_name_test: Optional[str]=None,
                        target_label: str = "Target (-)", show: Optional[bool] = False):
    errors_train = compute_differences(emulator, X_train, y_train)
    errors_test = compute_differences(emulator, X_test, y_test)
    rcp_name_to_list_of_years_and_errors_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(errors_train, errors_test, years_train, years_test, rcp_name_train, rcp_name_test, validation_mask)
    label = f'Error for {uncapitalize(target_label)}'
    plot_climatological_time_series(rcp_name_to_list_of_years_and_errors_and_color_and_label, errors_train, label, f'', show)

def compute_differences(emulator: PySREmulator, X: Optional[np.ndarray], y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if X is None:
        return None
    else:
        y_predicted = emulator.predict(X)
        errors = [y_predicted_value - y_value for y_value, y_predicted_value in zip(y, y_predicted)]
        return np.array(errors)






