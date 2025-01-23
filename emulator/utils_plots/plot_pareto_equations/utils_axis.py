import math
from typing import Callable, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.scale import FuncScale

from emulator.utils_metric.metric import Metric, metric_to_label


def set_x_axis(ax: Axes, x_ticks: list[int]):
    x_min, x_max = 0, max(x_ticks) + 1
    ax_twin = ax.twiny()
    ax.set_xlim((x_min, x_max))
    ax.set_xticks([])
    ax_twin.set_xlim((x_min, x_max))
    ax_twin.set_xticks(x_ticks)
    ax_twin.set_xlabel('Complexity')


def set_custom_y_axis(ax: Axes, loss_list: list[float], target_label: str, metric: Metric):
    """Scale y-axis with a log scale for large values then a linear scale for smaller values"""
    threshold = 2.0 * min(loss_list)
    ax.set_yscale(FuncScale(ax.yaxis, custom_functions_for_yaxis(threshold)))
    small_ticks = [float(t) / 10 for t in range(math.floor(10 * min(loss_list)), math.ceil(10 * threshold))][::2]
    large_ticks = [t * 10 for t in [1, 10, 100, 1000, 10000, 100000] if t < max(loss_list)]
    y_ticks = small_ticks + large_ticks
    ax.set_yticks(y_ticks)
    ax.set_ylim((y_ticks[0], y_ticks[-1]))
    ax.set_ylabel(f'{metric_to_label[metric]} for {target_label}')


def custom_functions_for_yaxis(threshold: float) -> tuple[Callable, Callable]:
    """Create custom functions for scaling axis: a log scale above a threshold, a linear scale below the threshold"""

    scale = 0.05

    def forward(value: np.ndarray) -> np.ndarray:
        return np.where(value <= threshold, value, threshold + scale * np.log(value / threshold))

    def inverse(value: np.ndarray) -> np.ndarray:
        return np.where(value <= threshold, value, threshold * np.exp((value - threshold) / scale))

    return forward, inverse
