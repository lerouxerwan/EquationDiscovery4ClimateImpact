from matplotlib import pyplot as plt

from projects.ensemble_30.df_for_ensemble_30 import get_df_for_ensemble_30
from utils.utils_plot import show_or_save_plot


def plot_npp_curves(show: bool):
    ax = plt.gca()
    for ensemble_id in list(range(1, 31))[:]:
        df = get_df_for_ensemble_30(ensemble_id)
        years = df.index.values
        values = df['NPP'].values
        ax.plot(years, values)

    plot_name = f'npp_curves'
    show_or_save_plot(plot_name, show)

if __name__ == '__main__':
    plot_npp_curves(show=True)
