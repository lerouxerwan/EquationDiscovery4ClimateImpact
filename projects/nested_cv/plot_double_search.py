from typing import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_double_search import OptimizationDoubleSearch
from optimization.utils_nested_cv import run_nested_cv
from optimization.utils_params.utils_params_values import ParamsValues
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot

NB_TOP_FAST = 6


def get_data(model_selection: str, fast: bool):
    data = OrderedDict()
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    nb_top = NB_TOP_FAST if fast else 10
    nb_top_ranges = list(range(1, nb_top + 1))[-1:]
    nb_top_ranges = [4, 5, 6, 7]
    for nb_top_hyperparameters in nb_top_ranges:
        log_info(f'Log for {nb_top_hyperparameters}')
        opt = OptimizationDoubleSearch(model_selection, ParamsValues.DEFAULT_CENTRED, nb_top_hyperparameters)
        rmse_test_list = run_nested_cv(dataset, opt, fast=False)
        data[nb_top_hyperparameters] = rmse_test_list
    df = pd.DataFrame(data)
    plot_name = f"compare_double_search_{len(rmse_test_list)}folds_for_{model_selection}"
    y_label = 'RMSE test'
    return df, plot_name, y_label


def get_data_diff(data_best, data_validated):
    df_best, *_ = data_best
    df_validated, *_ = data_validated
    df_diff = 100 * (df_validated - df_best) / df_best
    plot_name = f"compare_marginal_search_for_different_model_selection"
    label = "Improvement of 'validated' w.r.t 'best' (%)"
    return df_diff, plot_name, label

def plot_compare_search(fast: bool):
    data_best = get_data('best', fast)
    data_validated = get_data('validated', fast)
    # data_diff = get_data_diff(data_best, data_validated)
    for data in [data_best, data_validated]:
        _plot_compare_double_search(*data, fast)

def _plot_compare_double_search(df: pd.DataFrame, plot_name: str, label: str, fast: bool):
    df_plot = df[df.sum().sort_values().index]
    ax = plt.gca()
    sns.boxplot(ax=ax, data=df_plot.melt(), x="variable", y="value")
    ax.set_ylabel(label)
    ax.set_xlabel("#Top hyperparameters")
    plt.xticks(rotation=90)
    ax.set_xticklabels([str(i) for i in df_plot.columns])
    plt.grid(axis='y')
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    min_values, mean_values, max_values = df_plot.min().values, df_plot.mean().values, df_plot.max().values
    mean_values = [f'{round(me, 2)} ({round(mi, 2)}, {round(ma, 2)})'
                   for mi, me, ma in zip(min_values, mean_values, max_values)]
    ax_twin.set_xticklabels([str(mean_value) for mean_value in mean_values], rotation=90)
    show_or_save_plot(plot_name + '_' + str(len(df)), show=fast)

if __name__ == '__main__':
    plot_compare_search(fast=True)
