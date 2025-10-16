import math
from typing import Optional

from pysr.utils import ArrayLike

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.dataset.plot_dataset import plot_values_feature
from utils.utils_plot import show_and_save_with_optional_plot_folder, get_subplots


def plot_selected_features(emulator: Emulator, dataset:Dataset, show: Optional[bool] = False,
                            plot_folder: Optional[str] = None) -> None:
    """Plot the features in the selected equation"""
    selected_feature_indexes = get_selected_feature_indexes(emulator.selected_variable_names, dataset.X_variable_names)
    selected_feature_indexes = selected_feature_indexes[:2]
    ncols = 2
    nrows = max(1, math.ceil(len(selected_feature_indexes) / ncols))
    fig, axs = get_subplots(nrows=nrows, ncols=ncols, sharex=True)
    if nrows == 1:
        for selected_feature_index, ax in zip(selected_feature_indexes, axs):
            plot_values_feature(ax, dataset, selected_feature_index)
    else:
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                selected_feature_index = selected_feature_indexes[i * ncols + j]
                plot_values_feature(ax, dataset, selected_feature_index)
    show_and_save_with_optional_plot_folder(f'selected_features', show, plot_folder)

def get_selected_feature_indexes(selected_variable_names: list[str], variable_names: Optional[ArrayLike[str]] = None) -> list[int]:
    if variable_names is None:
        return [int(selected_variable_name[1:]) for selected_variable_name in selected_variable_names]
    else:
        return [variable_names.index(selected_variable_name) for selected_variable_name in selected_variable_names]

