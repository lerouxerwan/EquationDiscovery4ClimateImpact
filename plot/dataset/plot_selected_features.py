from typing import Optional

import matplotlib.pyplot as plt
from pysr.utils import ArrayLike

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.dataset.plot_dataset import plot_values_feature
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_selected_features(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False,
                            plot_folder: Optional[str] = None) -> None:
    """Plot the features in the selected equation"""
    for top_feature_index in get_selected_feature_indexes(emulator.selected_variable_names, dataset.X_variables_names):
        ax = plt.gca()
        plot_values_feature(ax, dataset, top_feature_index)
        show_and_save_with_optional_plot_folder(f'feature_#{top_feature_index}', show, plot_folder)

def get_selected_feature_indexes(selected_variable_names: list[str], variable_names: Optional[ArrayLike[str]] = None) -> list[int]:
    if variable_names is None:
        return [int(selected_variable_name[1:]) for selected_variable_name in selected_variable_names]
    else:
        return [variable_names.index(selected_variable_name) for selected_variable_name in selected_variable_names]

