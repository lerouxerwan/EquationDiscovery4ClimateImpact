from typing import Any

import matplotlib.pyplot as plt

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
    # Add a bar plot for the PySR score
    ax_twin = ax.twinx()
    ax_twin.bar(coordinate_list[-1], emulator.score_list, width=width, label='PySR score', color='blue')
    ax_twin.set_ylabel('PySR score')
    # Add a line for PySR threshold
    threshold_constant_values = [emulator.threshold_for_best_model_selection for _ in complexity_list]
    ax.plot(complexity_list, threshold_constant_values, color=split_name_to_color["train"],
            linestyle='--', label='Threshold for equation selection')
    # Add rounded equations on the X axis
    ax.set_xlabel('Equations with rounded coefficients\n(which may explain why the complexity seems wrong)')
    # General settings for the plot
    x_ticks = complexity_list
    set_x_axis(ax, x_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([get_equation_str(expr) for expr in emulator.expr_list], rotation=45, ha='right', rotation_mode='anchor')
    set_custom_y_axis(ax, loss_list, target_label, metric)
    ax.legend(loc='upper right')
    show_or_save_plot(f'loss_vs_complexity', show)

def load_bar_attributes(nb_bars: int, complexity_list: list[int]):
    assert all([c % 2 == 1 for c in complexity_list]), 'A case with pair complexity must be implemented'
    width = 2 / (1 + nb_bars) # add one for the blank bar
    coordinates_list = [[c  + width * (bar_id - nb_bars / 2 + 0.5) for c in complexity_list] for bar_id in range(nb_bars)]
    return width, coordinates_list




    # # For the train split, we show the equation near each point
    # if split_name == 'train':
    #     best_complexity = emulator.get_best()['complexity']
    #     for complexity, loss, expr in zip(complexity_list, loss_list, emulator.expr_list):
    #         bold = complexity == best_complexity
    #         equation_str = get_equation_str(expr)
    #         if bold:
    #             equation_str = '\n'.join(['$\\mathbf{' + s[1:-1] + '}$' for s in equation_str.split('\n')])
    #         ax.text(x=complexity, y=loss, s=equation_str, fontsize=FONTSIZE,
    #                 rotation=90, verticalalignment='bottom', horizontalalignment='left')
    # return loss_list





