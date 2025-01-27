from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.metric import Metric
from emulator.utils_plots.plot_pareto_equations.utils_axis import set_custom_y_axis, set_x_axis
from emulator.utils_plots.utils_equation_str import get_equation_str
from emulator.utils_plots.utils_plot_split_name import SPLIT_NAMES, split_name_to_color
from utils.utils_plot import show_or_save_plot


def plot_loss_vs_complexity(emulator: ClimateImpactEmulator, split_name_to_x_and_y: dict[str, Any],
                            target_label: str = "Target (-)", show: bool = False, metric=Metric.RMSE) -> None:
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
    add_bar_plot_for_PySR_score(ax, coordinate_list, emulator, width)
    # Add rounded equations on the lower X axis
    ax.set_xlabel('Equations with rounded coefficients\n(which may explain why the complexity seems wrong)')
    x_ticks = complexity_list
    set_x_axis(ax, x_ticks)
    ax.set_xticks(x_ticks)
    xticklabels = [get_equation_str(expr) for expr in emulator.expr_list]
    xticklabels[complexity_list.index(emulator.selected_complexity)] = get_equation_str(emulator.selected_expr, add_bold=True)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    # Add y axis with special scaling
    set_custom_y_axis(ax, loss_list, target_label, metric)
    plot_threshold(ax, emulator, metric, *ax.get_xlim())
    # General settings for the plot
    ax.legend(loc='upper right')
    show_or_save_plot(f'loss_vs_complexity', show)


def add_bar_plot_for_PySR_score(ax, coordinate_list, emulator, width):
    #  Add a bar plot for the PySR score
    ax_twin = ax.twinx()
    color_PySR_score = 'blue'
    ax_twin.bar(coordinate_list[-1], emulator.score_list, width=width, color=color_PySR_score)
    ax_twin_ymin, ax_twin_ymax = ax_twin.get_ylim()
    ax_twin.set_ylim(ax_twin_ymin, 2 * ax_twin_ymax)
    ax_twin.set_ylabel('PySR score', color=color_PySR_score)


def plot_threshold(ax, emulator, metric, xmax, xmin):
    # Add a line for PySR threshold
    if metric is Metric.MSE:
        constant_value = emulator.threshold_for_best_model_selection
    elif metric is Metric.RMSE:
        constant_value = np.sqrt(emulator.threshold_for_best_model_selection)
    else:
        raise NotImplementedError
    x_for_threshold = [xmin, xmax]
    threshold_constant_values = [constant_value for _ in x_for_threshold]
    ax.plot(x_for_threshold, threshold_constant_values, color=split_name_to_color["train"],
            linestyle='--', label='Threshold for equation selection')


def load_bar_attributes(nb_bars: int, complexity_list: list[int]):
    # assert all([c % 2 == 1 for c in complexity_list]), 'A case with pair complexity must be implemented'
    width = 1 / (1 + nb_bars) # add one for the blank bar
    coordinates_list = [[c  + width * (bar_id - nb_bars / 2 + 0.5) for c in complexity_list] for bar_id in range(nb_bars)]
    return width, coordinates_list





