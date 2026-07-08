from cProfile import label

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.ensemble_30.df_for_ensemble_30 import get_df_for_ensemble_30, get_mean_df_for_ensemble_30
from utils.utils_plot import show_or_save_plot


def plot_npp_curves(show: bool):
    ax = plt.gca()
    small_size = 2
    large_size = 3

    # Plot each line
    cmap = plt.get_cmap('Spectral')
    ensemble_ids = list(range(1, 31))[:]
    colors = [cmap(i) for i in np.linspace(0, 1, len(ensemble_ids))]
    for color, ensemble_id in zip(colors, ensemble_ids):
        df = get_df_for_ensemble_30(ensemble_id)
        ax.plot(df.index.values, df['NPP'].values, linestyle='-', color=color, marker='o', markersize=small_size * 2, linewidth=small_size)
    # Plot average line
    df = get_mean_df_for_ensemble_30()
    ax.plot(df.index.values, df['NPP'].values, label='Ensemble average', color='k', marker='o', markersize=large_size * 2, linewidth=large_size)

    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    ax.set_xlabel('Year')
    ax.set_ylabel(get_label(dataset.target_label))
    ax.legend()
    plot_name = f'npp_curves'
    show_or_save_plot(plot_name, show)

if __name__ == '__main__':
    plot_npp_curves(show=False)
