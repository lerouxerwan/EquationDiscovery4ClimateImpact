from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.utlis_metric_box import add_metric_box
from emulator.utils_plots.utils_equation_str import get_equation_str
from utils.utils_plot import show_or_save_plot


def plot_scatter(emulator: ClimateImpactEmulator, split_name_to_X_and_y: dict[str, Any], show: bool) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    for split_name, (X, y) in split_name_to_X_and_y.items():
        _plot_scatter(emulator, split_name, X, y, show)


def _plot_scatter(emulator: ClimateImpactEmulator, split_name: str, X, y, show: bool) -> None:
    ax = plt.gca()
    y_predicted = emulator.predict(X)
    ax.scatter(y, y_predicted)
    # Add grid and diagonal line
    ax.grid()
    all_y = np.append(y, y_predicted)
    lower_bound = min(all_y) - 1
    upper_bound = max(all_y) + 1
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color='grey', linestyle='--')
    # Annotate equation
    coef = 0.95
    ax.annotate(f'Equation found: {get_equation_str(emulator.selected_expr)}', xy=(0.02, 0.95),
                xycoords='axes fraction', textcoords='offset points', size=7,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))
    # Add metric box
    add_metric_box(ax, y, y_predicted, split_name.capitalize(), (0.5, 0.1))
    # Add legend and labels
    target_label = 'value'
    x_label, ylabel = f"True {target_label}", f"Predicted {target_label}"
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    # ax.legend(loc='center right')
    show_or_save_plot(f'plot_scatter_{split_name}', show)



