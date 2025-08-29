from typing import Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.utils_metric.utlis_metric_box import add_metric_box
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, get_ymin_and_ymax
from plot.by_split.utlis_plot_selected_equation import get_true_label_and_predicted_label, add_equation
from utils.utils_plot import show_and_save_with_optional_plot_folder, get_subplots


def plot_scatter(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    for split_name, (X, y, y_predicted, years) in split_name_to_X_and_y_and_y_predicted_and_years.items():
        fig, ax = plt.subplots()
        _plot_scatter(ax, dataset, emulator, fig, split_name, y, y_predicted, years, ymax, ymin)
        show_and_save_with_optional_plot_folder(f'plot_scatter_{split_name}', show, plot_folder)


def _plot_scatter(ax, dataset, emulator, fig, split_name, y, y_predicted, years, ymax, ymin):
    cmap = matplotlib.cm.viridis
    c = years
    ax.scatter(y, y_predicted, c=c, cmap=cmap)
    norm = matplotlib.colors.BoundaryNorm(c, cmap.N)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', label='Years')
    #  Add grid and diagonal line
    ax.grid()
    ax.plot([ymin, ymax], [ymin, ymax], color='grey', linestyle='--')
    # Annotate equation and metric box
    add_equation(ax, emulator.selected_expr)
    add_metric_box(ax, y, y_predicted, dataset.target_label, split_name)
    #  Add legend and labels
    x_label, y_label = get_true_label_and_predicted_label(dataset.target_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #  Set ranges of axis
    ax.set_xlim((ymin, ymax))
    ax.set_ylim((ymin, ymax))


def plot_scatter_side_by_side(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    fig, axs = get_subplots(1, 2)

    # Plot first axis
    _, y_train, y_predicted_train, years_train = split_name_to_X_and_y_and_y_predicted_and_years['train']
    _, y_validation, y_predicted_validation, years_validation = split_name_to_X_and_y_and_y_predicted_and_years['validation']
    y, y_predicted, years = [], [], []
    i_train, i_validation = 0, 0
    n_train, n_validation = len(y_train), len(y_validation)
    while (i_train < n_train) or (i_validation < n_validation):
        if (i_train < n_train) and ((i_validation == n_validation) or (years_train[i_train] < years_validation[i_validation])):
            y.append(y_train[i_train])
            y_predicted.append(y_predicted_train[i_train])
            years.append(years_train[i_train])
            i_train += 1
        else:
            y.append(y_validation[i_validation])
            y_predicted.append(y_predicted_validation[i_validation])
            years.append(years_validation[i_validation])
            i_validation += 1
    y, y_predicted, years = np.array(y), np.array(y_predicted), np.array(years)
    _plot_scatter(axs[0], dataset, emulator, fig, "Train + Validation", y, y_predicted, years, ymax, ymin)

    # Plot second axis
    split_name = 'test'
    _, y, y_predicted, years = split_name_to_X_and_y_and_y_predicted_and_years[split_name]
    _plot_scatter(axs[1], dataset, emulator, fig, split_name, y, y_predicted, years, ymax, ymin)

    show_and_save_with_optional_plot_folder(f'plot_scatter_side_by_side', show, plot_folder)






