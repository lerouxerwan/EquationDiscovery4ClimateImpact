from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from plot.by_rcp.utils_rcp import rcp_name_to_color, get_rcp_label
from plot.by_split.utils_plot_by_split import set_default_years


def load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train: np.ndarray,
                                                             y_test: Optional[np.ndarray] = None,
                                                             years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str='RCP85',
                                                             rcp_name_test: Optional[str]=None, nb_historical_years: Optional[int] = None):
    rcp_name_to_list_of_years_and_y_and_color = dict()
    # Some checks
    assert y_train.ndim == 1
    assert (years_train is None) or (years_train.ndim == 1)
    # Cast all y as ndarray (instead of Series) if it is not already done
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
        if y_test is not None:
            y_test = y_test.values
    # Set default for years_train and years_test if needed
    years_test, years_train = set_default_years(y_test, y_train, years_test, years_train)
    # Set value of nb_historical_years
    start_rcp_index = 0 if nb_historical_years is None else nb_historical_years
    # Add rcp_name_train
    rcp_name_to_list_of_years_and_y_and_color[rcp_name_train] = [
        (years_train[:start_rcp_index], y_train[:start_rcp_index], 'k', 'Historical'),
        (years_train[start_rcp_index:], y_train[start_rcp_index:], rcp_name_to_color[rcp_name_train], get_rcp_label(rcp_name_train))
    ]
    # Add rcp_name_test
    if rcp_name_test is not None:
        rcp_name_to_list_of_years_and_y_and_color[rcp_name_test] = [
            (years_train[:start_rcp_index], y_train[:start_rcp_index], 'k', None),
            (years_test, y_test, rcp_name_to_color[rcp_name_test], get_rcp_label(rcp_name_test))
        ]
    # Some final checks
    for list_of_years_and_y_and_color_and_label in rcp_name_to_list_of_years_and_y_and_color.values():
        for years, y, *_ in list_of_years_and_y_and_color_and_label:
            assert len(years) == len(y)
    return rcp_name_to_list_of_years_and_y_and_color

def plot_average_value(ax: Axes, color: str, values: np.ndarray, dates: np.ndarray, window_size: int = 30, plot_std: bool = True):
    assert len(values) == len(dates)
    #  Plot average value as line
    # Only plot a line if the number of years is larger than the window size
    if len(dates) > window_size:
        shift = window_size // 2
        window_values_list = [values[i - shift: i + shift] for i in range(shift, len(dates) - shift)]
        averaged_values = [np.mean(window_values) for window_values in window_values_list]
        years_average = dates[shift:-shift]
        ax.plot(years_average, averaged_values, color=color)
        if plot_std:
            # By default, we consider the default std, i.e. the biased one with ddof = 0
            half_std_values = [np.std(window_values, ddof=1) / 2 for window_values in window_values_list]
            averaged_values, half_std_values = np.array(averaged_values), np.array(half_std_values)
            ax.fill_between(years_average, averaged_values - half_std_values, averaged_values + half_std_values, color=color, alpha=0.4)
            std_values = [2.*v for v in half_std_values]
        else:
            std_values = None
        return years_average, std_values

