from cProfile import label

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.ensemble_30.df_for_ensemble_30 import get_df_for_ensemble_30, get_mean_df_for_ensemble_30
from utils.utils_plot import show_or_save_plot


def plot_npp_curves(show: bool):
    ax = plt.gca()
    # Plot each line
    for ensemble_id in list(range(1, 31))[:]:
        df = get_df_for_ensemble_30(ensemble_id)
        ax.plot(df.index.values, df['NPP'].values, linestyle='', marker='o', markersize=2)
    # Plot average line
    df = get_mean_df_for_ensemble_30()
    ax.plot(df.index.values, df['NPP'].values, label='Ensemble average', color='grey', linewidth=4)

    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    ax.set_xlabel('Year')
    ax.set_ylabel(get_label(dataset.target_label))
    ax.legend()
    plot_name = f'npp_curves'
    show_or_save_plot(plot_name, show)

if __name__ == '__main__':
    plot_npp_curves(show=False)
