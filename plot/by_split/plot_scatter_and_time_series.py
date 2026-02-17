from typing import Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from plot.by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from plot.by_split.plot_scatter import _plot_scatter
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, get_ymin_and_ymax
from plot.by_split.utlis_plot_selected_equation import get_true_and_predicted_label, get_label
from plot.utils_metric.utlis_metric_box import add_metric_box
from utils.utils_plot import show_or_save_plot, get_subplots


def plot_scatter_and_time_series(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False) -> None:
    """Plot a 2x2 plot"""

    fig, all_axs = get_subplots(2, 2)





    # FIRST PLOT
    axs = all_axs[0, :]
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test,
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    ymax += 3

    split_name_to_rcp_name = {
        'Train + validation': 'RCP85',
        'Test': 'RCP45',
    }
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(dataset.y_train, dataset.y_test, dataset.years_train, dataset.years_test, dataset.rcp_name_train, dataset.rcp_name_test, dataset.nb_historical_years)


    for j, (ax, (split_name, rcp_name)) in enumerate(zip(axs, split_name_to_rcp_name.items())):

        list_of_years_and_y_and_color_and_label = rcp_name_to_list_of_years_and_y_and_color_and_label[rcp_name]
        if rcp_name == 'RCP45':
            list_of_years_and_y_and_color_and_label = list_of_years_and_y_and_color_and_label[1:]
        y_list = []
        y_predicted_list = []
        years_list = []
        for years, y, color, label in list_of_years_and_y_and_color_and_label:
            years = [int(year) for year in years]
            if rcp_name == 'RCP85':
                years_train_to_index = {y: i for i, y in enumerate(dataset.years_train)}
                indexes = [years_train_to_index[year] for year in years]
                X = dataset.X_train[indexes, :]
            else:
                years_test_to_index = {y: i for i, y in enumerate(dataset.years_test)}
                indexes = [years_test_to_index[year] for year in years]
                X = dataset.X_test[indexes, :]
            y_predicted = emulator.predict(X)
            y_predicted_list.append(y_predicted)
            y_list.append(y)
            years_list.append(years)
            #  Add label
            y_true_label, y_predicted_label = get_true_and_predicted_label()
            common_kwargs = {'linestyle': '', 'color': color}
            ax.plot(years, y, marker='o', **common_kwargs)
            ax.plot(years, y_predicted, marker='*', **common_kwargs)
            # Annotate equation and metric box
            #  Add legend and labels
            ax.set_xlabel('Years')
            ax.set_ylabel(get_label(dataset.target_label))
            ax.legend(loc='upper right')
            ax.set_ylim((ymin, ymax))
        # Plot average
        window_size = 30
        shift = window_size // 2
        dates = np.concat(years_list)
        years_average = dates[shift:-shift]


        window_values_list = [np.concat(y_list)[i - shift: i + shift] for i in range(shift, len(dates) - shift)]
        ax.plot(years_average,  [np.mean(window_values) for window_values in window_values_list], color='gray')

        window_values_list = [np.concat(y_predicted_list)[i - shift: i + shift] for i in range(shift, len(dates) - shift)]
        linestyle_prediction = 'dashed'
        ax.plot(years_average, [np.mean(window_values) for window_values in window_values_list], color='gray', linestyle=linestyle_prediction)
        # Add metric box
        add_metric_box(ax, np.concat(y_list), np.concat(y_predicted_list), dataset.target_label, split_name,
                       x_and_y_location=(0.05, 0.05))

        #  Add a first legend to explain the dot and the line
        legend_labels = ['Reference', 'Prediction', f'{window_size}-years average for the reference', f'{window_size}-years average for the prediction']
        legend_handles = [
            plt.Line2D([0], [0], marker='o', linestyle='', color='grey', markerfacecolor='w'),
            plt.Line2D([0], [0], marker='*', linestyle='', color='grey', markerfacecolor='w'),
            plt.Line2D([0], [0], marker='', linestyle='-', color='grey'),
            plt.Line2D([0], [0], marker='', linestyle=linestyle_prediction, color='grey'),

        ]
        ax.legend(legend_handles, legend_labels, loc='upper right', ncol=2)
        letter = 'ab'[j]
        ax.text(0.1, 0.92, f'({letter})', weight="bold", fontsize=10, transform=ax.transAxes)
        #  Add a second legend for the color only if needed
    # add_equation(emulator.selected_equation)

    # SECOND PLOT
    axs = all_axs[1, :]

    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator,
                                                                                                           dataset.X_train,
                                                                                                           dataset.y_train,
                                                                                                           dataset.X_test,
                                                                                                           dataset.y_test,
                                                                                                           dataset.years_train,
                                                                                                           dataset.years_test,
                                                                                                           dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)

    #  Plot first axis
    X_train, y_train, y_predicted_train, years_train = split_name_to_X_and_y_and_y_predicted_and_years['train']
    if dataset.validation_split is ValidationSplit.NONE:
        X, y, y_predicted, years = X_train, y_train, y_predicted_train, years_train
    else:
        X_validation, y_validation, y_predicted_validation, years_validation = \
        split_name_to_X_and_y_and_y_predicted_and_years['validation']
        X, y, y_predicted, years = [], [], [], []
        i_train, i_validation = 0, 0
        n_train, n_validation = len(y_train), len(y_validation)
        while (i_train < n_train) or (i_validation < n_validation):
            if (i_train < n_train) and (
                    (i_validation == n_validation) or (years_train[i_train] < years_validation[i_validation])):
                X.append(X_train[i_train])
                y.append(y_train[i_train])
                y_predicted.append(y_predicted_train[i_train])
                years.append(years_train[i_train])
                i_train += 1
            else:
                X.append(X_validation[i_validation])
                y.append(y_validation[i_validation])
                y_predicted.append(y_predicted_validation[i_validation])
                years.append(years_validation[i_validation])
                i_validation += 1
        X, y, y_predicted, years = np.array(X), np.array(y), np.array(y_predicted), np.array(years)
    X_test, y_test, y_predicted_test, years_test = split_name_to_X_and_y_and_y_predicted_and_years['test']
    all_years = sorted(list(set(years).union(set(years_test))))
    vmin_and_vmax = min(all_years), max(all_years)
    cmap = matplotlib.cm.viridis
    # cmap = matplotlib.cm.gist_rainbow

    #  Plot first axis
    _plot_scatter(axs[0], dataset, emulator, fig, "Train + Validation", X, y, y_predicted, years, ymax, ymin,
                  add_colorbar=False, cmap=cmap, vmin_and_vmax=vmin_and_vmax, letter='c')

    #  Plot second axis
    _plot_scatter(axs[1], dataset, emulator, fig, 'test', X_test, y_test, y_predicted_test, years_test, ymax, ymin,
                  add_colorbar=False, cmap=cmap, vmin_and_vmax=vmin_and_vmax, letter='d')

    norm = matplotlib.colors.BoundaryNorm(all_years, cmap.N)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='horizontal', label='Years',
                 pad=-0.4, fraction=0.065)

    print(emulator.selected_equation)
    all_axs[0, 0].set_title(f"Diagnosis of the linear equation: {emulator.selected_equation}", loc='left', pad=20)
    show_or_save_plot(f'plot_scatter_and_time_series', show)












