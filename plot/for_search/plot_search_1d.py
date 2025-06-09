from itertools import combinations
from typing import Optional, Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from data.utils_experiment.experiment import Experiment
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME, \
    get_cv_results_column_name, RMSE_VALIDATION_COLUMN_NAME, RMSE_TRAIN_COLUMN_NAME
from plot.by_split.utils_axis import set_ylabel_with_metric
from plot.utils_metric.metric import Metric
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_diagnosis_search_1d(experiment: Experiment, param_grid: dict[str, list], target_label: str,
                             show: bool, plot_folder: Optional[str] = None) -> None:
    """Plot the variation of RMSE validation for each hyperparameter in the param_grid"""
    params_list = experiment.df_cv_results[PARAMS_EMULATOR_COLUMN_NAME].to_list()
    for param_name in param_grid.keys():
        ax = plt.gca()
        params_values = [params[param_name] for params in params_list]
        for model_selection in ['best', 'custom']:
            _plot_search_1d(ax, experiment.df_cv_results, params_values, model_selection)
        ax.set_xlabel(' '.join([w.capitalize() for w in param_name.split('_')]))
        set_ylabel_with_metric(ax, Metric.RMSE, target_label)
        ax.legend()
        show_and_save_with_optional_plot_folder(f"search_1D/{param_name}", show, plot_folder)


def _plot_search_1d(ax: Axes, df_cv_results: pd.DataFrame, param_values: list[float], model_selection: str) -> None:
    rmse_train_column_name = get_cv_results_column_name(model_selection, RMSE_TRAIN_COLUMN_NAME)
    rmse_validation_column_name = get_cv_results_column_name(model_selection, RMSE_VALIDATION_COLUMN_NAME)
    min_rmse_validation_list = []
    corresponding_rmse_train_list = []
    sorted_param_values = sorted(list(set(param_values)))
    for sorted_param_value in sorted_param_values:
        ind = pd.Series(index=df_cv_results.index, data=[v == sorted_param_value for v in param_values])
        # Compute min rmse validation and the corresponding rmse train
        index_min_rmse_validation = df_cv_results.loc[ind, rmse_validation_column_name].idxmin()
        min_rmse_validation_list.append(df_cv_results.loc[index_min_rmse_validation, rmse_validation_column_name])
        corresponding_rmse_train_list.append(df_cv_results.loc[index_min_rmse_validation, rmse_train_column_name])
    ax.plot(sorted_param_values, min_rmse_validation_list, label=rmse_validation_column_name.replace('_', ' '), marker='o')
    ax.plot(sorted_param_values, corresponding_rmse_train_list, label=rmse_train_column_name.replace('_', ' '), marker='o')





