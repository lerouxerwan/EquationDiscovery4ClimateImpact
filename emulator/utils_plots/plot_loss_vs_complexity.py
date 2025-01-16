from typing import Any

import matplotlib.pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_plots.utils_equation_str import get_equation_str
from emulator.utils_plots.utils_axis import set_custom_y_axis, set_x_axis
from emulator.utils_plots.utils_plot_split_name import SPLIT_NAMES, split_name_to_linestyle, \
    split_name_to_marker
from utils.utils_plot import show_or_save_plot, FONTSIZE


def plot_loss_vs_complexity(emulator: ClimateImpactEmulator, split_name_to_x_and_y: dict[str, Any], show: bool) -> None:
    """Plot prediction loss as a function of complexity for several splits
    Note that for the train split it will correspond to the pareto front"""
    ax = plt.gca()
    # One plot for each split
    loss_list = []
    for split_name, (X, y) in split_name_to_x_and_y.items():
        loss_list.extend(_plot_loss_vs_complexity(ax, emulator, X, y, split_name))
    # General settings for the plot
    set_x_axis(ax, emulator.maxsize)
    set_custom_y_axis(ax, loss_list, y_units=emulator.y_units_)
    ax.legend(loc='lower left', fontsize=FONTSIZE)
    show_or_save_plot(f'loss_vs_complexity', show)


def _plot_loss_vs_complexity(ax, emulator: ClimateImpactEmulator, X, y, split_name: str) -> list[float]:
    """Plot prediction loss as a function of complexity for a single split"""
    assert emulator.equations_ is not None, 'emulator has not been fitted'
    assert split_name in SPLIT_NAMES, f'{split_name} is not in {SPLIT_NAMES}'
    # Plot loss versus complexity
    complexity_list = emulator.complexity_list
    loss_list = emulator.compute_loss(X, y)
    ax.plot(complexity_list, loss_list, label=split_name,
            marker=split_name_to_marker[split_name], color='k', linestyle=split_name_to_linestyle[split_name])
    # For the train split, we show the equation near each point
    if split_name == 'train':
        best_complexity = emulator.get_best()['complexity']
        for complexity, loss, expr in zip(complexity_list, loss_list, emulator.expr_list):
            bold = complexity == best_complexity
            equation_str = get_equation_str(expr)
            if bold:
                equation_str = '\n'.join(['$\\mathbf{' + s[1:-1] + '}$' for s in equation_str.split('\n')])
            ax.text(x=complexity, y=loss, s=equation_str, fontsize=FONTSIZE,
                    rotation=90, verticalalignment='bottom', horizontalalignment='left')
    return loss_list





