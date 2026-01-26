from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import lineStyles

from data.utils_dataset.dataset import Dataset
from emulator.emulator import Emulator
from plot.by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from plot.by_split.utils_plot_split_name import SPLIT_NAMES
from plot.utils_metric.utlis_metric_box import add_metric_box
from plot.by_split.utils_plot_by_split import load_split_name_to_X_and_y_and_y_predicted_and_years, \
    get_ymin_and_ymax
from plot.by_split.utlis_plot_selected_equation import add_equation, get_label, get_true_and_predicted_label
from utils.utils_plot import show_or_save_plot, get_subplots


def plot_time_series_side_by_side(emulator: Emulator, dataset: Dataset, show: Optional[bool] = False) -> None:
    """Plot predicted values VS True values (as 2 time series)"""
    split_name_to_X_and_y_and_y_predicted_and_years = load_split_name_to_X_and_y_and_y_predicted_and_years(emulator, dataset.X_train, dataset.y_train, dataset.X_test, 
                                                       dataset.y_test, dataset.years_train, dataset.years_test, dataset.validation_mask)
    ymin, ymax = get_ymin_and_ymax(split_name_to_X_and_y_and_y_predicted_and_years)
    ymax += 3
    fig, axs = get_subplots(1, 2)
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
    add_equation(emulator.selected_equation)

    show_or_save_plot(f'time_series_side_by_side', show)