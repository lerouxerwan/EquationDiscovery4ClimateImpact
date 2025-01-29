from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from emulator.climate_impact_emulator import ClimateImpactEmulator
from emulator.utils_metric.utlis_metric_box import add_metric_box
from emulator.utils_plots.plot_by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, \
    get_ymin_and_ymax
from emulator.utils_plots.plot_by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label, \
    add_equation, get_label
from utils.utils_plot import show_or_save_plot


def plot_time_series(emulator: ClimateImpactEmulator,X_train: np.ndarray | pd.DataFrame,
                                         y_train: np.ndarray | pd.Series,
                                         X_test: Optional[np.ndarray | pd.DataFrame]=None,
                                         y_test: Optional[np.ndarray | pd.Series]=None,
                     years_train: Optional[np.ndarray]=None, years_test: Optional[np.ndarray]=None,
                     target_label: str = "Target (-)", show: bool = False) -> None:
    """Plot predicted values VS True values (as 2 time series)"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, X_train, y_train, X_test, y_test, years_train, years_test)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        ax = plt.gca()
        #  Add grid on Y-axis and two lines
        ax.yaxis.grid()
        y_true_label, y_predicted_label = get_true_label_and_predicted_label(target_label, remove_units=True)
        ax.plot(years, y, label=y_true_label)
        ax.plot(years, y_predicted, label=y_predicted_label)
        # Annotate equation and metric box
        add_equation(ax, emulator.selected_expr)
        add_metric_box(ax, y, y_predicted, target_label, split_name)
        #  Add legend and labels
        ax.set_xlabel('Years')
        ax.set_ylabel(get_label(target_label))
        ax.legend(loc='lower left')
        ax.set_ylim((ymin, ymax))
        show_or_save_plot(f'plot_time_series_{split_name}', show)