import numpy as np
from matplotlib.axes import Axes

from data.utils_dataset.dataset import Dataset
from plot.by_rcp.plot_time_series_climato import plot_climatological_time_series_function
from plot.by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from numpy import ndarray

def plot_values_feature(ax: Axes, dataset: Dataset, feature_index: int):
    plot_values(ax, dataset, dataset.X_train[:, feature_index], dataset.X_test[:, feature_index], dataset.X_labels[feature_index])


def plot_values_target(ax: Axes, dataset: Dataset):
    plot_values(ax, dataset, dataset.y_train, dataset.y_test, dataset.y_labels[0])


def plot_values(ax: Axes, dataset: Dataset, values_train: ndarray, values_test: ndarray, label: str):
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(
        values_train, values_test, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test,
        dataset.nb_historical_years)
    plot_climatological_time_series_function(ax, rcp_name_to_list_of_years_and_y_and_color_and_label, values_train, label,
                                             None)
    y_all = np.concat([values_train, values_test], axis=0)
    ymin, ymax = 0.99 * np.min(y_all), 1.01 * np.max(y_all)
    ax.set_ylim(ymin, ymax)