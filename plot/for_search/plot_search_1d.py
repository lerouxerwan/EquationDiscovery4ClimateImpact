from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from data.utils_experiment.experiment import Experiment
from emulator.utils_hyperparameter_search.utils_column_names import PARAMS_EMULATOR_COLUMN_NAME
from utils.utils_plot import show_and_save_with_optional_plot_folder


def plot_diagnosis_search_1d(experiment: Experiment, show: bool, plot_folder: Optional[str] = None):
    """Plot the variation of RMSE validation for each hyperparameter in the param_grid"""
    df = experiment.df_cv_results
    params_list = df[PARAMS_EMULATOR_COLUMN_NAME].to_list()
    metric_name = experiment.rmse_val_column_name
    for param_name in experiment.get_combinations_of_search_param_names(nb_elements=1):
        param_name = param_name[0]
        ax = plt.gca()
        min_loss_list = []
        param_values = [params[param_name] for params in params_list]
        sorted_param_values = sorted(list(set(param_values)))
        for sorted_param_value in sorted_param_values:
            ind = pd.Series(index=df.index, data=[v == sorted_param_value for v in param_values])
            min_loss = df.loc[ind, metric_name].min()
            min_loss_list.append(min_loss)
        ax.plot(sorted_param_values, min_loss_list)
        create_label = lambda s: ' '.join([w.capitalize() for w in s.split('_')])
        ax.set_xlabel(create_label(param_name))
        ax.set_ylabel(create_label(metric_name))
        show_and_save_with_optional_plot_folder(f"diagnosis_1D_{param_name}", show, plot_folder)

