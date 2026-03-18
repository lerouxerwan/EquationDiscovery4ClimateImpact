import math
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.scale import FuncScale

from plot.utils_metric.metric import Metric, metric_to_label
from plot.by_split.utlis_plot_selected_equation import get_unit
from numpy import ndarray

def set_x_axis(ax: Axes, x_ticks: list[int]):
    x_min, x_max = 0, max(x_ticks) + 1
    ax_twin = ax.twiny()
    ax.set_xlim((x_min, x_max))
    ax.set_xticks([])
    ax_twin.set_xlim((x_min, x_max))
    ax_twin.set_xticks(x_ticks)
    ax_twin.set_xlabel('Complexity')


def set_log_y_axis(ax: Axes, loss_list: list[float], target_label: str, metric: Metric):
    """Scale y-axis with a log scale for large values then a linear scale for smaller values"""
    min_pow = math.floor(math.log10(min(loss_list)))
    max_pow = math.ceil(math.log10(max(loss_list)))
    y_ticks = [math.pow(10, i) for i in range(min_pow, max_pow + 1)]
    threshold = y_ticks[0]
    ax.set_yscale(FuncScale(ax.yaxis, functions_for_yaxis(threshold)))
    ax.set_yticks(y_ticks)
    ax.set_ylim((y_ticks[0], y_ticks[-1]))
    set_ylabel_with_metric(ax, metric, target_label)


def set_ylabel_with_metric(ax, metric, target_label):
    unit = '(-)' if metric is Metric.NLL else get_unit(target_label)
    ax.set_ylabel(f'{metric_to_label[metric]} {unit}')


def functions_for_yaxis(threshold: float) -> tuple[Callable, Callable]:
    """Create functions for scaling axis: a log scale above a threshold, a linear scale below the threshold"""

    scale = 0.05

    def forward(value: ndarray) -> ndarray:
        return np.where(value <= threshold, value, threshold + scale * np.log(value / threshold))

    def inverse(value: ndarray) -> ndarray:
        return np.where(value <= threshold, value, threshold * np.exp((value - threshold) / scale))

    return forward, inverse
