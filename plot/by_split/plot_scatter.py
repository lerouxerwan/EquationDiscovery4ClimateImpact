from typing import Optional, Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
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


def _plot_scatter(ax, dataset, emulator, fig, split_name, y, y_predicted, years, ymax, ymin,
                  add_colorbar: bool = True, cmap = None, vmin_and_vmax = None):
    if cmap is None:
        cmap = matplotlib.cm.viridis
    c = years
    if vmin_and_vmax is None:
        ax.scatter(y, y_predicted, c=c, cmap=cmap)
    else:
        vmin, vmax = vmin_and_vmax
        ax.scatter(y, y_predicted, c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    if add_colorbar:
        norm = matplotlib.colors.BoundaryNorm(c, cmap.N)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', label='Years',
                     pad=0.2)
    #  Add grid and diagonal line
    ax.grid()
    ax.plot([ymin, ymax], [ymin, ymax], color='grey', linestyle='--')
    # Annotate equation and metric box
    add_equation(ax, emulator.selected_expr)
    add_metric_box(ax, y, y_predicted, dataset.target_label, split_name)
    # Add second metric box
    nb_years = dataset.nb_historical_years
    y_reference_future, y_predicted_future, years_future = y[-nb_years:], y_predicted[-nb_years:], years[-nb_years:]
    y_reference_historical, years_historical = dataset.y_train[:nb_years], dataset.years_train[:nb_years]
    y_predicted_historical = emulator.predict(dataset.X_train[:nb_years, :])
    delta_reference = compute_delta(y_reference_historical, y_reference_future)
    delta_predicted = compute_delta(y_predicted_historical, y_predicted_future)
    difference = delta_predicted - delta_reference
    text_to_annotate = f'Change between {years_historical[0]}-{years_historical[-1]} and {years_future[0]}-{years_future[-1]}:\n'
    text_to_annotate += f'Reference = {round(delta_reference, 2)}%, '
    text_to_annotate += f'Predicted = {round(delta_predicted, 2)}%'
    # text_to_annotate += f'Difference = {round(difference, 2)}%'
    coef = 0.95
    x_and_y_location = (0.03, 0.85)
    ax.annotate(text_to_annotate, xy=x_and_y_location, xycoords='axes fraction', textcoords='offset points',
                xytext=x_and_y_location,
                bbox=dict(boxstyle="round", fc=(coef, coef, coef), ec="none"))
    #  Add legend and labels
    x_label, y_label = get_true_label_and_predicted_label(dataset.target_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #  Set ranges of axis
    ax.set_xlim((ymin, ymax))
    ax.set_ylim((ymin, ymax))

def compute_delta(y_historical, y_future) -> float:
    mean_historical, mean_future = np.mean(y_historical), np.mean(y_future)
    return 100 * (mean_future - mean_historical) / mean_historical


def plot_scatter_side_by_side(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False, plot_folder: Optional[str] = None) -> None:
    """Plot predicted values VS True values (in a scattered way) side by side"""

    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    fig, axs = get_subplots(1, 2)

    # Plot first axis
    _, y_train, y_predicted_train, years_train = split_name_to_X_and_y_and_y_predicted_and_years['train']
    if dataset.validation_split is ValidationSplit.NONE:
        y, y_predicted, years = y_train, y_predicted_train, years_train
    else:
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
    _, y_test, y_predicted_test, years_test = split_name_to_X_and_y_and_y_predicted_and_years['test']
    all_years = sorted(list(set(years).union(set(years_test))))
    vmin_and_vmax = min(all_years), max(all_years)
    cmap = matplotlib.cm.viridis
    # cmap = matplotlib.cm.gist_rainbow

    # Plot first axis
    _plot_scatter(axs[0], dataset, emulator, fig, "Train + Validation", y, y_predicted, years, ymax, ymin,
                  add_colorbar=False, cmap=cmap, vmin_and_vmax=vmin_and_vmax)

    # Plot second axis
    _plot_scatter(axs[1], dataset, emulator, fig, 'test', y_test, y_predicted_test, years_test, ymax, ymin,
                  add_colorbar=False, cmap=cmap, vmin_and_vmax=vmin_and_vmax)

    norm = matplotlib.colors.BoundaryNorm(all_years, cmap.N)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='horizontal', label='Years',
                 pad=-0.25, fraction=0.065)

    show_and_save_with_optional_plot_folder(f'plot_scatter_side_by_side', show, plot_folder)






