import math
from typing import Optional

from data.utils_dataset.npp_season_v1 import get_dataset
from plot.dataset.plot_dataset import plot_values_feature
from utils.utils_plot import get_subplots, show_or_save_plot


def main_four_features(show: Optional[bool] = False) -> None:
    """Plot the features in the selected equation"""
    dataset =  get_dataset()
    selected_names = ['SST_DJF', 'Shortwave_DJF', 'SST_MAM', 'SSS_MAM']
    selected_feature_indexes = [dataset.X_variable_names.index(selected_name) for selected_name in selected_names]

    for index in selected_feature_indexes[::2]:
        dataset.X_test[:, index] -= 273.15
        dataset.X_train[:, index] -= 273.15
        dataset.X_labels[index] = dataset.X_labels[index].replace('(K)', '($^o$C)')

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
    show_or_save_plot(f'four_features', show)


if __name__ == '__main__':
    main_four_features(show=False)