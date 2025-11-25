from collections import OrderedDict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from utils.utils_plot import show_or_save_plot


def main_plot_correlation_feature(show: bool = False):
    dataset = get_dataset(validation_size=0., validation_split=ValidationSplit.NONE)
    d = OrderedDict()
    for variable_name in sorted(dataset.X_variable_names):
        i = dataset.X_variable_names.index(variable_name)
        x = dataset.X_train[:, i]
        d[variable_name] = x
    df = pd.DataFrame.from_dict(d)

    # Visualisation of one global matrix
    ax = plt.gca()
    sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', center=0, annot=False, fmt=".2f")
    plt.title("Correlation matrix for the features")
    show_or_save_plot(f'correlation_matrix', show)

    # Visualisation of one matrix for each variable
    for i in range(23):
        df_local = df.iloc[:, i*5:(i+1)*5]
        ax = plt.gca()
        sns.heatmap(df_local.corr(), ax=ax, cmap='coolwarm', center=0, annot=True, fmt=".2f")
        plt.title("Correlation matrix for the features")
        show_or_save_plot(f'correlation_matrix_variable_{i}', show)

    # Visualisation of one matrix for each season/time moment
    for i in range(5):
        df_local = df.iloc[:, i::5]
        ax = plt.gca()
        sns.heatmap(df_local.corr(), ax=ax, cmap='coolwarm', center=0, annot=False, fmt=".2f")
        plt.title("Correlation matrix for the features")
        show_or_save_plot(f'correlation_matrix_season_{i}', show)


if __name__ == '__main__':
    main_plot_correlation_feature(show=False)