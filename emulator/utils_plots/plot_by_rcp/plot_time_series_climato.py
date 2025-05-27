import math
from typing import Optional, Any

import numpy as np
from matplotlib import pyplot as plt

from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import plot_average_value
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_label
from utils.utils_plot import show_or_save_plot, show_and_save_with_optional_plot_folder


def plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label: dict[str, list[tuple[list[int], list[float], str, str]]],
                                    y_train: np.ndarray, y_label: str, plot_name: str, show: Optional[bool], ymin_and_ymax: tuple[float, float] = None, 
                                    plot_std: bool = True, plot_folder: Optional[str] = None) -> dict[str, tuple[list[int], list[float], str]]:
    ax = plt.gca()
    rcp_name_to_years_and_std_values_and_color = _plot_climatological_time_series(ax,
                                                                                  rcp_name_to_list_of_years_and_y_and_color_and_label,
                                                                                  y_train, y_label,
                                                                                  ymin_and_ymax, plot_std)
    show_and_save_with_optional_plot_folder(f'climatological_series_{plot_name}', show, plot_folder)
    return rcp_name_to_years_and_std_values_and_color


def _plot_climatological_time_series(ax, rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, y_label,
                                     ymin_and_ymax, plot_std: bool = True, plot_average: bool = True):
    window_size = 30
    all_dates = []
    rcp_name_to_years_and_std_values_and_color = {}
    for rcp_name, list_of_y_and_years_and_color_and_label in rcp_name_to_list_of_years_and_y_and_color_and_label.items():
        dates, values = [], []
        #  Plot for each sub period the points in their respective color
        for years, y, color, label in list_of_y_and_years_and_color_and_label:
            ax.plot(years, y, color=color, linestyle='', marker='o', label=label)
            values.append(y)
            dates.append(years)
        dates, values = np.concat(dates), np.concat(values)
        #  Plot the average mean/std with the last color, i.e. the color of the RCP,
        if plot_average:
            years_average, std_values = plot_average_value(ax, color, values, dates, window_size, plot_std)
            rcp_name_to_years_and_std_values_and_color[rcp_name] = (years_average, std_values, color)
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
    if ymin_and_ymax is not None:
        assert len(ymin_and_ymax) == 2
        ax.set_ylim(ymin_and_ymax)
    ax.set_ylabel(y_label)
    #  Add first legend
    increasing_trend = (y_train[0] < y_train[-1]) if isinstance(y_train, np.ndarray) else (
            y_train.values[0] < y_train.values[-1])
    loc1, loc2 = ('upper left', 'lower right') if increasing_trend else ('upper right', 'lower left')
    ax.legend(loc=loc1)
    #  Add a second legend to explain the dot and the line
    legend_labels = ['Annual value', f'{window_size}-years average', 'Standard deviation']
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w'),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=10,
                   alpha=0.5),
    ]
    if not plot_average:
        legend_handles, legend_labels = legend_handles[:1], legend_labels[:1]
    elif not plot_std:
        legend_handles, legend_labels = legend_handles[:-1], legend_labels[:-1]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    ax_twin.legend(legend_handles, legend_labels, loc=loc2)
    ax.yaxis.grid()
    return rcp_name_to_years_and_std_values_and_color
