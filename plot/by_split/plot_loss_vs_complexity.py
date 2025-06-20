from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.utils_metric.metric import Metric
from plot.by_split.utils_axis import set_log_y_axis, set_x_axis
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y
from plot.by_split.utils_equation_str import get_equation_str
from plot.by_split.utils_plot_split_name import SPLIT_NAMES, split_name_to_color, \
    get_label_split_name
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_loss_vs_complexity(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False,
                            plot_folder: Optional[str] = None) -> None:
    """Plot prediction loss as a function of complexity for several splits
    Note that for the train split it will correspond to the pareto front"""
    detailed_plot = False
    metric = Metric.RMSE
    split_name_to_x_and_y = load_split_name_to_X_and_y(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    fig, ax = plt.subplots(figsize=(16, 9))
    complexity_list = emulator.complexity_list
    # Detailed plot adds one bar for PySR score
    nb_bars = len(split_name_to_x_and_y) + int(detailed_plot)
    width, coordinate_list = load_bar_attributes(nb_bars=nb_bars, complexity_list=complexity_list)
    # One bar plot for each split
    all_loss_list = []
    sorted_split_names = [split_name for split_name in SPLIT_NAMES if split_name in split_name_to_x_and_y]
    for bar_id, split_name in enumerate(sorted_split_names):
        X, y = split_name_to_x_and_y[split_name]
        coordinates = coordinate_list[bar_id]
        loss_list = emulator.compute_loss_list_other_metric(X, y, metric=metric)
        # Filter values where the loss is equal np.nan
        coordinates, loss_list = list(zip(*[(coordinate, loss) for coordinate, loss in zip(coordinates, loss_list) if not np.isnan(loss)]))
        barplot = ax.bar(coordinates, loss_list, width=width, label=get_label_split_name(split_name, dataset),
               color=split_name_to_color[split_name])
        loss_list_labels = [str(round(loss, 2)) for loss in loss_list]
        ax.bar_label(barplot, labels=loss_list_labels, label_type='edge', padding=1)
        all_loss_list.extend(loss_list)

    # Add rounded equations on the lower X axis
    ax.set_xlabel('Equations with rounded coefficients')
    x_ticks = complexity_list
    set_x_axis(ax, x_ticks)
    ax.set_xticks(x_ticks)
    # Add equations as ticklabels (show in bold the selected equation)
    xticklabels = [get_equation_str(expr) for expr in emulator.expr_list]
    xticklabels[complexity_list.index(emulator.selected_complexity)] = get_equation_str(emulator.selected_expr, add_bold=True)
    # Add y-axis with special scaling
    set_log_y_axis(ax, all_loss_list, dataset.target_label, metric)
    # Potentially add detailed plots
    if detailed_plot:
        add_bar_plot_for_PySR_score(ax, coordinate_list, emulator, width)
        plot_threshold(ax, emulator, metric, *ax.get_xlim())
    # General settings for the plot
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper right')
    show_and_save_with_optional_plot_folder('loss_vs_complexity', show, plot_folder)



def add_bar_plot_for_PySR_score(ax, coordinate_list, emulator, width):
    #  Add a bar plot for the PySR score
    ax_twin = ax.twinx()
    color_PySR_score = 'blue'
    ax_twin.bar(coordinate_list[-1], emulator.score_list, width=width, color=color_PySR_score)
    ax_twin_ymin, ax_twin_ymax = ax_twin.get_ylim()
    ax_twin.set_ylim(ax_twin_ymin, 2 * ax_twin_ymax)
    ax_twin.set_ylabel('PySR score', color=color_PySR_score)


def plot_threshold(ax, emulator, metric, xmax, xmin):
    if emulator.model_selection == 'best':
        threshold = 1.5
        # Add a line for PySR threshold
        if metric is Metric.MSE:
            constant_value = threshold
        elif metric is Metric.RMSE:
            constant_value = np.sqrt(threshold)
        else:
            raise NotImplementedError
        x_for_threshold = [xmin, xmax]
        threshold_constant_values = [constant_value for _ in x_for_threshold]
        ax.plot(x_for_threshold, threshold_constant_values, color=split_name_to_color["train"],
                linestyle='--', label='Threshold for equation selection')


def load_bar_attributes(nb_bars: int, complexity_list: list[int]):
    # assert all([c % 2 == 1 for c in complexity_list]), 'A case with pair complexity must be implemented'
    width = (2 if nb_bars == 2 else 1) / (1 + nb_bars) # add one for the blank bar
    coordinates_list = [[c  + width * (bar_id - nb_bars / 2 + 0.5) for c in complexity_list] for bar_id in range(nb_bars)]
    return width, coordinates_list





