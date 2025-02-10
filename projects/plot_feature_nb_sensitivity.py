import os
from operator import itemgetter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset import load_dataset_dataframe
from data.utils_search import get_dataset_search_path, CSV_FILENAME, METRIC_COLUMN_NAME
import os.path as op

from utils.utils_plot import show_or_save_plot


def plot_feature_nb_sensitivity(X, y, validation_size: float = 0.3, feature_selection_name: str = 'PySRDefault',
                                show: bool = False):
    # Load param_name to a sorted list
    dataset_search_path = get_dataset_search_path(X, y, validation_size)
    param_folder_to_list_nb_features_and_filepath = dict()
    for experiment_folder in os.listdir(dataset_search_path):
        if experiment_folder.startswith(feature_selection_name):
            nb_features = int(experiment_folder.split("_")[-1])
            for param_folder in os.listdir(op.join(dataset_search_path, experiment_folder)):
                filepath_search_result = op.join(dataset_search_path, experiment_folder, param_folder, CSV_FILENAME)
                value = (nb_features, filepath_search_result)
                if param_folder in param_folder_to_list_nb_features_and_filepath:
                    param_folder_to_list_nb_features_and_filepath[param_folder].append(value)
                else:
                    param_folder_to_list_nb_features_and_filepath[param_folder] = [value]
    # Sort the param_name_list
    param_folder_to_list_nb_features_and_filepath = {param_folder: list(sorted(v, key=itemgetter(0)))
                                                     for param_folder, v in param_folder_to_list_nb_features_and_filepath.items()}
    # Create plot
    ax = plt.gca()
    colors = ['blue', "green", "grey"]
    for j, (label, list_nb_features_and_filepath) in enumerate(param_folder_to_list_nb_features_and_filepath.items()):
        color = colors[j]
        # list_nb_features_and_filepath = list_nb_features_and_filepath[1:]
        nb_features_list = [nb_features for nb_features, _ in list_nb_features_and_filepath]
        mse_series_list = [-pd.read_csv(filepath)[METRIC_COLUMN_NAME] for _, filepath in list_nb_features_and_filepath]
        rmse_series_list = [mse_series.apply(np.sqrt) for mse_series in mse_series_list]
        best_rmse = [float(rmse_series.iloc[0]) for rmse_series in rmse_series_list]
        median_rmse = [rmse_series.median() for rmse_series in rmse_series_list]
        ax.plot(nb_features_list, best_rmse, label=label, marker='x', color=color)
        ax.plot(nb_features_list, median_rmse, label=None, marker='o', color=color, linestyle=':')
    ax.legend(loc='upper right')
    ax.set_xlabel('Number of selected features')
    ax.set_ylabel('RMSE validation')
    #  Add a second legend to explain the dot and the line
    legend_labels = ['Best RMSE', 'Average RMSE']
    legend_handles = [
        plt.Line2D([0], [0], marker='x', linestyle='', color='k', markerfacecolor='w'),
        plt.Line2D([0], [0], marker='o', linestyle='-', color='k'),
    ]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    ax_twin.legend(legend_handles[::-1], legend_labels[::-1], loc='upper left')
    ax.yaxis.grid()
    show_or_save_plot("feature_nb_sensitivity", show)


def main_ranking(filename):
    X_train, y_train,  *_ = load_dataset_dataframe(filename)
    plot_feature_nb_sensitivity(X_train, y_train, show=True)

if __name__ == '__main__':
    main_ranking("NPP_season.csv")
    main_ranking("NPP_season_25_variables.csv")
