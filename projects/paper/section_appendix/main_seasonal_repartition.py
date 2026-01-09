import os.path as op

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from utils.utils_path import DATA_PATH
from utils.utils_plot import get_subplots, show_or_save_plot

folder = 'NPPz_season'
periods = ['HIST_20', 'RCP45_94', 'RCP85_94']
period_labels = ['Historical (1986-2005)',
                 'RCP4.5 (2006-2099)',
                 'RCP8.5 (2006-2099)']
period_to_label = dict(zip(periods, period_labels))


def main_plot_seasonal_repartition_three_pies(show: bool = True):
    fig, axs = get_subplots(1, 3, wspace=-0.5, hspace=0)
    for j, (ax, period) in enumerate(zip(axs, periods)):
        plot_seasonal_repartition_one_pie(ax, period, j)
    show_or_save_plot('seasonal_repartition', show)

def plot_seasonal_repartition_one_pie(ax: Axes, period: str, j: int):
    filepath = op.join(DATA_PATH, folder, f'{folder}_GOL4_allDepths_{period}_mean.csv')
    labels = ['winter', 'spring', 'summer', 'autumn']
    abbreviations = ['DJF', 'MAM', 'JJA', 'SON']
    colors = ['blue', 'green', 'yellow', 'brown']
    df = pd.read_csv(filepath, index_col=0)
    for column, abbreviation in zip(df.columns, abbreviations):
        assert column.endswith(abbreviation)
    values = df.sum().values
    values *= 100 / sum(values)
    ax.set_title(period_to_label[period])
    ax.pie(values, labels=None, colors=colors, autopct='%1.1f%%')
    if j == 1:
        ax.legend(labels, ncol=4, loc='lower center')


if __name__ == '__main__':
    main_plot_seasonal_repartition_three_pies(show=False)