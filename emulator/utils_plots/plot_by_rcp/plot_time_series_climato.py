import math

import numpy as np
from matplotlib import pyplot as plt

from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import plot_average_value
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_label
from utils.utils_plot import show_or_save_plot


def plot_climatological_time_series(rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, prefix,
                                    show, ymin_and_ymax: tuple[float, float] = None):
    ax = plt.gca()
    window_size = 30
    all_dates = []
    rcp_name_to_std_values_and_years_and_color = {}
    for rcp_name, list_of_y_and_years_and_color_and_label in rcp_name_to_list_of_years_and_y_and_color_and_label.items():
        dates, values = [], []
        #  Plot for each sub period the points in their respective color
        for years, y, color, label in list_of_y_and_years_and_color_and_label:
            ax.plot(years, y, color=color, linestyle='', marker='o', label=label)
            values.append(y)
            dates.append(years)
        dates, values = np.concat(dates), np.concat(values)
        #  Plot the average mean/std with the last color, i.e. the color of the RCP,
        years_average, std_values = plot_average_value(ax, color, values, dates, window_size)
        rcp_name_to_std_values_and_years_and_color[rcp_name] = (std_values, years_average, color)
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
        ax.set_ylim(ymin_and_ymax)
    ax.set_ylabel(f'{prefix} {get_label(target_label)}')
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
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    ax_twin.legend(legend_handles, legend_labels, loc=loc2)
    ax.yaxis.grid()
    show_or_save_plot(f'climatological_series_{prefix}', show)
    return rcp_name_to_std_values_and_years_and_color
