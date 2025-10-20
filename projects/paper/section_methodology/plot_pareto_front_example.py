import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.scale import FuncScale

from emulator.emulator import Emulator
from plot.by_split.utils_axis import functions_for_yaxis
from plot.utils_metric.metric import Metric
from utils.utils_plot import show_or_save_plot


def plot_pareto_front_example(emulator: Emulator, X: np.ndarray, y: np.ndarray,
                              target_label: str = "Target (-)", show: bool = False) -> None:
    """Plot prediction loss as a function of complexity for several splits
    Note that for the train split it will correspond to the pareto front"""
    ax = plt.gca()
    complexity_list = emulator.complexity_list
    width, coordinate_list = load_bar_attributes(nb_bars=2, complexity_list=complexity_list)
    ax.bar(coordinate_list[0], emulator.loss_list, width=width, color='red')
    ax.set_xlabel('Equation f')
    # Add rounded equations on the lower X axis
    x_ticks = complexity_list
    set_x_axis_example(ax, x_ticks)
    ax.set_xticks(x_ticks)
    xticklabels = [equation.replace('x0', 'x') for equation in emulator.equation_list]
    # xticklabels[complexity_list.index(emulator.selected_complexity)] = get_equation_str(emulator.selected_expr, add_bold=True).replace('x0', 'x')
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    # Add y-axis with special scaling
    set_log_y_axis_example(ax, emulator.loss_list, target_label, emulator.metric_)
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


def set_log_y_axis_example(ax: Axes, loss_list: list[float], target_label: str, metric: Metric):
    """Scale y-axis with a log scale for large values then a linear scale for smaller values"""
    ax.set_yscale(FuncScale(ax.yaxis, functions_for_yaxis(3.1)))
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





