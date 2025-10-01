from typing import OrderedDict

import pandas as pd

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_nested_cv import run_nested_cv
import seaborn as sns
import matplotlib.pyplot as plt

from optimization.utils_params.utils_params_values import param_names, ParamsValues
from utils.utils_plot import show_or_save_plot

NB_PARAMS_FAST = 2


def get_data(model_selection: str, fast: bool):
    data = OrderedDict()
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    param_names_for_plot = param_names[:NB_PARAMS_FAST] if fast else param_names
    for param_name in param_names_for_plot:
        opt = OptimizationMarginalSearch(model_selection, ParamsValues.DEFAULT_CENTRED, param_name)
        rmse_test_list = run_nested_cv(dataset, opt, fast)
        data[param_name] = rmse_test_list
    df = pd.DataFrame(data)
    plot_name = f"compare_marginal_search_{len(rmse_test_list)}folds_for_{model_selection}"
    y_label = 'RMSE test'
    return df, plot_name, y_label


def get_data_diff(data_best, data_validated):
    df_best, *_ = data_best
    df_validated, *_ = data_validated
    df_diff = 100 * (df_validated - df_best) / df_best
    plot_name = f"compare_marginal_search_for_different_model_selection"
    label = "Improvement of 'validated' w.r.t 'best' (%)"
    return df_diff, plot_name, label

def plot_compare_marginal_search(fast: bool):
    data_best = get_data('best', fast)
    data_validated = get_data('validated', fast)
    data_diff = get_data_diff(data_best, data_validated)
    for data in [data_best, data_validated, data_diff]:
        _plot_compare_marginal_search(*data, fast)

def _plot_compare_marginal_search(df: pd.DataFrame, plot_name: str, label: str, fast: bool):
    if fast:
        tops = [NB_PARAMS_FAST]
    else:
        tops = [5, 10, 20, 30]
    for top in tops:
        df_plot = df[df.sum().sort_values().index[:top]]
        ax = plt.gca()
        sns.boxplot(ax=ax, data=df_plot.melt(), x="variable", y="value")
        ax.set_ylabel(label)
        ax.set_xlabel("Hyperparameter")
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(ax.get_xticks())
        min_values, mean_values, max_values = df_plot.min().values, df_plot.mean().values, df_plot.max().values
        mean_values = [f'{round(me, 2)} ({round(mi, 2)}, {round(ma, 2)})'
                       for mi, me, ma in zip(min_values, mean_values, max_values)]
        ax_twin.set_xticklabels([str(mean_value) for mean_value in mean_values], rotation=90)
        show_or_save_plot(plot_name + '_' + str(top), show=fast)

if __name__ == '__main__':
    plot_compare_marginal_search(fast=False)
