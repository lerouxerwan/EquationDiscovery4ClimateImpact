import math
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from utils.utils_plot import show_or_save_plot


def plot_climatological_series(y_train: np.ndarray | pd.Series, y_test: Optional[np.ndarray | pd.Series] = None,
                               years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None, rcp_name_train: str= 'RCP85',
                               rcp_name_test: Optional[str]=None, nb_historical_years: int = 0,
                               target_label: str = "Target (-)", suffix: str = 'Observed', show: bool = False):
    """Plot several RCP climatological series on the same graph"""
    window_size = 30
    rcp_name_to_list_of_years_and_y_and_label_and_color = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, nb_historical_years)
    ax = plt.gca()
    all_dates = []
    for rcp_name, list_of_y_and_years_and_color_and_label in rcp_name_to_list_of_years_and_y_and_label_and_color.items():
        dates, values = [], []
        #  Plot for each sub period the points in their respective color
        for years, y, color, label in list_of_y_and_years_and_color_and_label:
            ax.plot(years, y, color=color, linestyle='', marker='o', label=label)
            values.append(y)
            dates.append(years)
        dates, values = np.concat(dates), np.concat(values)
        # Plot the average mean/std with the last color, i.e. the color of the RCP,
        plot_average_value(ax, color, values, dates, window_size)
        all_dates.append(dates)
    #  Set custom X-axis
    xmin = int(math.floor(np.min(np.concat(all_dates)) / 10.0)) * 10
    xmax = int(math.ceil(np.max(np.concat(all_dates)) / 10.0)) * 10
    ax.set_xlim(xmin, xmax)
    xticks = [x for x in range(xmin, xmax + 1, 10)]
    xticks_half = xticks[::2]
    ax.set_xticks(xticks_half if ((xticks_half[0] == xticks[0]) and (xticks_half[-1] == xticks[-1])) else xticks)
    ax.set_xlabel('Years')
    #  Y axis
    ax.set_ylabel(f'{suffix} {target_label.lower()}')

    #  Add first legend
    increasing_trend = (y_train[0] < y_train[-1]) if isinstance(y_train, np.ndarray) else (y_train.values[0] < y_train.values[-1])
    loc1, loc2 = ('upper left', 'lower right') if increasing_trend else ('upper right', 'lower left')
    ax.legend(loc=loc1)
    #  Add a second legend to explain the dot and the line
    legend_labels = ['Annual indicator', f'{window_size}-years average', 'Standard deviation']
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w'),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=10,
                   alpha=0.5),
    ]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    ax_twin.legend(legend_handles, legend_labels, loc=loc2)
    ax.yaxis.grid()

    show_or_save_plot(f'climatological_series_{suffix}', show)


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
