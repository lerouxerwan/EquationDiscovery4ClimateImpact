from typing import Optional

import numpy as np
import pandas as pd

from data.utils_dataset.dataset import Dataset
from emulator.pysr_emulator import PySREmulator
from emulator.utils_metric.metric import Metric, metric_to_function, metric_to_label
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label, \
    get_true_and_predicted_prefix, uncapitalize, get_label, get_true_and_predicted_label
from utils.utils_log import log_info
from utils.utils_plot import compute_axis_lim


def plot_climato(emulator: PySREmulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    y_train_predicted = emulator.predict(dataset.X_train)
    y_test_predicted = None if dataset.X_test is None else emulator.predict(dataset.X_test)
    y_values = np.concat([y for y in [dataset.y_train, dataset.y_test, y_train_predicted, y_test_predicted] if y is not None])
    ymin_and_ymax = compute_axis_lim(y_values)
    # Plot observation and prediction using the same limit
    true_target_label, predicted_target_label = get_true_label_and_predicted_label(dataset.target_label)
    rcp_name_to_years_and_observed_std_values_years_and_color = _plot_climato(dataset.y_train, dataset.y_test, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test,
                  dataset.validation_mask, true_target_label, show, ymin_and_ymax, plot_folder)
    rcp_name_to_years_and_predicted_std_values_and_color = _plot_climato(y_train_predicted, y_test_predicted, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test,
                  dataset.validation_mask, predicted_target_label, show, ymin_and_ymax, plot_folder)
    # Plot ratio of std
    rcp_name_to_list_of_years_and_y_and_color_and_label = {}
    for rcp_name, (years, observed_std_values, color) in rcp_name_to_years_and_observed_std_values_years_and_color.items():
        years, predicted_std_values, color = rcp_name_to_years_and_predicted_std_values_and_color[rcp_name]
        values = [predicted_v / observed_v for observed_v, predicted_v in zip(observed_std_values, predicted_std_values)]
        true_label, predicted_label = get_true_and_predicted_label()
        metric = 'std 30-years'
        prefix = f'{true_label} {metric} / {predicted_label} {metric}'
        rcp_name_to_list_of_years_and_y_and_color_and_label[rcp_name] = [(years, values, color, prefix)]
    y_label = f'{prefix}\nfor {uncapitalize(get_label(dataset.target_label))}'
    plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, dataset.y_train, y_label,
                                    "Std", show, plot_std=False, plot_folder=plot_folder)



def _plot_climato(y_train: np.ndarray, y_test: Optional[np.ndarray] = None,
                  years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                  rcp_name_test: Optional[str]=None, validation_mask:Optional[np.ndarray[bool]] = None,
                  target_label: str = "Target (-)", show: bool = False, ymin_and_ymax: Optional[tuple[float, float]] = None,
                  plot_folder: Optional[str] = None):
    """Plot several RCP climatological time series on the same graph"""
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, validation_mask)
    y_label = get_label(target_label)
    plot_name = y_label.split()[0]
    return plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, y_label, plot_name, show, ymin_and_ymax,
                                           plot_folder=plot_folder)


def plot_errors_climato(emulator: PySREmulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None):
    for relative_error in [True, False]:
        _plot_errors_climato(emulator, dataset, show, relative_error, plot_folder)


def _plot_errors_climato(emulator: PySREmulator, dataset: Dataset, show: Optional[bool] = False, relative_error: bool = False,
                         plot_folder: Optional[str] = None):
    errors_train = compute_differences(emulator, dataset.X_train, dataset.y_train, relative_error)
    errors_test = compute_differences(emulator, dataset.X_test, dataset.y_test, relative_error)
    rcp_name_to_list_of_years_and_errors_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(errors_train, errors_test, dataset.years_train, dataset.years_test,
                                                                                                                        dataset.rcp_name_train, dataset.rcp_name_test, dataset.validation_mask)
    target_label = uncapitalize(get_label(dataset.target_label))
    if relative_error:
        y_label = f'Relative error of {target_label.split('(')[0]}(%)'
    else:
        y_label = f'Error of {target_label}'
    plot_name = y_label.split()[0]
    plot_climatological_time_series(rcp_name_to_list_of_years_and_errors_and_color_and_label, errors_train, y_label, plot_name, show, plot_folder=plot_folder)

def compute_differences(emulator: PySREmulator, X: Optional[np.ndarray], y: Optional[np.ndarray],
                        relative_error: bool = False) -> Optional[np.ndarray]:
    if X is None:
        return None
    else:
        y_predicted = emulator.predict(X)
        if relative_error:
            errors = [100 * (y_predicted_value - y_value) / y_value for y_value, y_predicted_value in zip(y, y_predicted)]
            log_info(f'Max absolute relative difference: {max(errors)}')
        else:
            errors = [y_predicted_value - y_value for y_value, y_predicted_value in zip(y, y_predicted)]
        return np.array(errors)






