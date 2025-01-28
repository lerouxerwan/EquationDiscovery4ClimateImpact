from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.utlis_metric_box import add_metric_box
from emulator.utils_plots.plot_by_split.utils_plot_by_split import load_split_name_to_X_and_y, \
    load_split_name_to_X_and_y_and_y_predicted_and_years, get_ymin_and_ymax
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_labels, add_equation
from utils.utils_plot import show_or_save_plot


def plot_scatter(emulator: ClimateImpactEmulator, X_train: np.ndarray | pd.DataFrame,
                                         y_train: np.ndarray | pd.Series,
                                         X_test: Optional[np.ndarray | pd.DataFrame]=None,
                                         y_test: Optional[np.ndarray | pd.Series]=None,
                 years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None,
                 target_label: str = "Target (-)", show: bool = False) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, X_train, y_train, X_test, y_test, years_train, years_test)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.viridis
        c = years
        ax.scatter(y, y_predicted, c=c, cmap=cmap)
        norm = matplotlib.colors.BoundaryNorm(c, cmap.N)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', label='Years')
        # Add grid and diagonal line
        ax.grid()
        ax.plot([ymin, ymax], [ymin, ymax], color='grey', linestyle='--')
        # Annotate equation and metric box
        add_equation(ax, emulator.selected_expr)
        add_metric_box(ax, y, y_predicted, target_label, split_name)
        # Add legend and labels
        x_label, y_label = get_labels(target_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # Set ranges of axis
        ax.set_xlim((ymin, ymax))
        ax.set_ylim((ymin, ymax))
        show_or_save_plot(f'plot_scatter_{split_name}', show)







