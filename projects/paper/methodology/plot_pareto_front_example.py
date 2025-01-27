import math
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.scale import FuncScale

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.metric import Metric, metric_to_label
from emulator.utils_plots.plot_pareto_equations.utils_axis import set_custom_y_axis, set_x_axis, \
    custom_functions_for_yaxis
from emulator.utils_plots.utils_equation_str import get_equation_str
from emulator.utils_plots.utils_plot_split_name import SPLIT_NAMES, split_name_to_color
from utils.utils_plot import show_or_save_plot


def plot_pareto_front_example(emulator: ClimateImpactEmulator, split_name_to_x_and_y: dict[str, Any],
                            target_label: str = "Target (-)", show: bool = False, metric=Metric.MSE) -> None:
    """Plot prediction loss as a function of complexity for several splits
    Note that for the train split it will correspond to the pareto front"""
    ax = plt.gca()
    complexity_list = emulator.complexity_list
    nb_bars = 1 + len(split_name_to_x_and_y)
    width, coordinate_list = load_bar_attributes(nb_bars=nb_bars, complexity_list=complexity_list)
    # One bar plot for each split
    loss_list = []
    valid_split_names = [split_name for split_name in SPLIT_NAMES if split_name in split_name_to_x_and_y]
    for bar_id, split_name in enumerate(valid_split_names):
        X, y = split_name_to_x_and_y[split_name]
        coordinates = coordinate_list[bar_id]
        loss = emulator.compute_loss(X, y, metric=metric)
        ax.bar(coordinates, loss, width=width,
               label=split_name, color=split_name_to_color[split_name])
        loss_list.extend(loss)
    print(loss_list)
    ax.set_xlabel('Equation f')
    # Add rounded equations on the lower X axis
    x_ticks = complexity_list
    set_x_axis_example(ax, x_ticks)
    ax.set_xticks(x_ticks)
    xticklabels = [get_equation_str(expr).replace('x0', 'x') for expr in emulator.expr_list]
    xticklabels[complexity_list.index(emulator.selected_complexity)] = get_equation_str(emulator.selected_expr, add_bold=True).replace('x0', 'x')
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    # Add y axis with special scaling
    set_custom_y_axis_example(ax, loss_list, target_label, metric)
    # General settings for the plot
    # ax.legend(loc='upper right')
    show_or_save_plot(f'pareto_front_example', show)

def set_x_axis_example(ax: Axes, x_ticks: list[int]):
    x_min, x_max = 0, max(x_ticks) + 1
    ax_twin = ax.twiny()
    ax.set_xlim((x_min, x_max))
    ax.set_xticks([])
    ax_twin.set_xlim((x_min, x_max))
    ax_twin.set_xticks(x_ticks)
    ax_twin.set_xlabel('Complexity c(f)')


def set_custom_y_axis_example(ax: Axes, loss_list: list[float], target_label: str, metric: Metric):
    """Scale y-axis with a log scale for large values then a linear scale for smaller values"""
    ax.set_yscale(FuncScale(ax.yaxis, custom_functions_for_yaxis(3.1)))
    small_ticks = [0., 1.0, 2.0, 3.0]
    large_ticks = [t * 10 for t in [1, 10, 100, 1000, 10000, 100000, 1_000_000][::2] if t < max(loss_list)]
    y_ticks = small_ticks + large_ticks
    ax.set_yticks(y_ticks)
    ax.set_ylim((y_ticks[0], y_ticks[-1]))
    ax.set_ylabel(f'Empirical error l(f)')
    plt.ticklabel_format(scilimits=(-5, 8))
    # ax.ticklabel_format(useOffset=False)


def load_bar_attributes(nb_bars: int, complexity_list: list[int]):
    width = 2 / (1 + nb_bars) # add one for the blank bar
    coordinates_list = [[c for c in complexity_list] for bar_id in range(nb_bars)]
    return width, coordinates_list





