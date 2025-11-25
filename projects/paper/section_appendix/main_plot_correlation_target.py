from collections import OrderedDict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from utils.utils_plot import show_or_save_plot


def main_plot_correlation_target(show: bool = False):
    dataset = get_dataset(validation_size=0., validation_split=ValidationSplit.NONE)
    d = OrderedDict()
    d['target(t)'] = dataset.y_train[:-1]
    d['target(t+1)'] = dataset.y_train[1:]
    df = pd.DataFrame.from_dict(d)

    # Visualisation of the correlation with different metrics
    for method in ['pearson', 'kendall', 'spearman']:
        ax = plt.gca()
        df_corr = df.corr(method=method)
        print(method, round(df_corr.iloc[0, 1], 2))
        sns.heatmap(df_corr, ax=ax, cmap='coolwarm', center=0, annot=False, fmt=".2f")
        plt.title("Correlation matrix for two successive target")
        show_or_save_plot(f'correlation_matrix_{method}', show)



if __name__ == '__main__':
    main_plot_correlation_target(show=False)