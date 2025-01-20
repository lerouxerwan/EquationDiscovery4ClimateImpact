from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.utlis_metric_box import add_metric_box
from emulator.utils_plots.plot_selected_equation.utlis_plot_selected_equation import get_labels, add_equation
from utils.utils_plot import show_or_save_plot


def plot_scatter(emulator: ClimateImpactEmulator, split_name_to_X_and_y: dict[str, Any],
                 target_label: str = "Target (-)", show: bool = False) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    for split_name, (X, y) in split_name_to_X_and_y.items():
        ax = plt.gca()
        y_predicted = emulator.predict(X)
        ax.scatter(y, y_predicted)
        # Add grid and diagonal line
        ax.grid()
        all_y = np.append(y, y_predicted)
        lower_bound = min(all_y) - 1
        upper_bound = max(all_y) + 1
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color='grey', linestyle='--')
        # Annotate equation and metric box
        add_equation(ax, emulator.selected_expr)
        add_metric_box(ax, y, y_predicted, split_name.capitalize(), (0.5, 0.1))
        # Add legend and labels
        x_label, y_label = get_labels(target_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        show_or_save_plot(f'plot_scatter_{split_name}', show)







