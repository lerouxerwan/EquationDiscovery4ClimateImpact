from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.axes import Axes

from data.utils_dataset.utils_dataset_values import load_dataset_values
from data.utils_dataset.validation_split import ValidationSplit
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import _plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label


@dataclass
class Dataset(object):
    csv_filename: str
    rcp_name_train: str
    rcp_name_test: Optional[str] = None,
    validation_size: float = 0.3
    validation_split: ValidationSplit = ValidationSplit.RANDOM

    def __post_init__(self):
        self.values = load_dataset_values(self.csv_filename, self.rcp_name_train, self.rcp_name_test, self.validation_size, self.validation_split)
        (self.X_train, self.y_train, self.X_test, self.y_test, self.years_train, self.years_test,
         self.X_units, self.y_units, self.X_labels, self.y_labels, self.X_variables_names, self.y_variable_names,
         self.validation_mask) = self.values

    def plot_values_feature(self, ax: Axes, feature_index: int):
        self.plot_values(ax, self.X_train[:, feature_index], self.X_test[:, feature_index], self.X_labels[feature_index])

    def plot_values(self, ax: Axes, values_train: np.ndarray, values_test: np.ndarray, label: str):
        rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(
            values_train, values_test, self.years_train, self.years_test, self.rcp_name_train, self.rcp_name_test, self.validation_mask)
        _plot_climatological_time_series(ax, rcp_name_to_list_of_years_and_y_and_color_and_label, values_train, label,
                                         None)
        y_all = np.concat([values_train, values_test], axis=0)
        ymin, ymax = 0.99 * np.min(y_all), 1.01 * np.max(y_all)
        ax.set_ylim(ymin, ymax)
