from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.scenarios.utils_scenarios import get_color, get_marker, get_years_and_values, \
    get_color_universal, get_scenario_label
from utils.utils_plot import show_or_save_plot


def main_plot_rcp_and_ssp(variable_name: str, plot_anomaly: bool = True, plot_std: bool = True, show: bool = False):
    # Display parameters
    ax = plt.gca()
    window_size = 30
    markersize = 5
    linewidth = 3
    start_reference_year = 1986
    end_reference_year = 2014

    # Combinations to show
    model_to_scenarios = OrderedDict()
    # model_to_scenarios['RCSM4'] = ['RCP45', 'RCP85']
    model_to_scenarios['RCSM6B'] = ['SSP370', 'SSP585'][:1]
    # model_to_scenarios['RCSM6'] = ['SSP585']

    for model, scenarios in model_to_scenarios.items():

        # Model attributes
        marker_model = get_marker(model)

        # Extract the average reference value for the model
        historical_years, historical_values = get_years_and_values(model, 'HIST', variable_name)
        historical_year_to_value = dict(zip(historical_years, historical_values))
        reference_years = range(start_reference_year, end_reference_year)
        reference_values = [historical_year_to_value[year] for year in reference_years]
        average_reference_value = np.mean(reference_values)

        # Plot the historical anomalies
        historical_plot_values = historical_values - average_reference_value if plot_anomaly else historical_values
        ax.plot(historical_years, historical_plot_values, label=None, color='k',
                linestyle='', marker=marker_model, markersize=markersize)

        for scenario in scenarios:
            # Scenario attributes
            color = get_color(model, scenario)

            # Plot scenario anomalies
            scenario_years, scenario_values = get_years_and_values(model, scenario, variable_name)
            scenario_plot_values = scenario_values - average_reference_value if plot_anomaly else scenario_values
            ax.plot(scenario_years, scenario_plot_values, label=None, color=color,
                    linestyle='', marker=marker_model, markersize=markersize)

            # Plot average anomaly
            all_years = np.concatenate([historical_years, scenario_years])
            all_plot_values = np.concatenate([historical_plot_values, scenario_plot_values])
            plot_average_value(ax, color, all_plot_values, all_years, window_size, plot_std, linewidth, marker_model,
                               3 * markersize)

    # Axes labels
    ax.set_xlabel('Year')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    if variable_name == 'NPP':
        y_label = get_label(dataset.target_label)
    else:
        y_label = dataset.X_labels[dataset.X_variable_names.index(variable_name)]
    if plot_anomaly:
        y_label = y_label[0].lower() + y_label[1:]
        y_label, unit = y_label.split(' (')
        y_label = (f'Anomaly of {y_label}\n'
                   f'with respect to the reference period {start_reference_year}-{end_reference_year} ({unit}')

    ax.set_ylabel(y_label)

    # First legend
    label_to_color = OrderedDict()
    label_to_color['Historical'] = 'k'
    for scenarios in model_to_scenarios.values():
        for scenario in scenarios:
            label_to_color[get_scenario_label(scenario)] = get_color_universal(scenario)
    first_legend_labels =  list(label_to_color.keys())
    first_legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                       color=color, markerfacecolor=color, markersize=markersize) for color in label_to_color.values()]
    loc = 'upper left' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower left'
    ax.legend(first_legend_handles, first_legend_labels, loc=loc)

    # Second legend
    second_legend_labels =  ['Annual value', f'{window_size}-years average', 'Standard deviation']
    second_legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w', markersize=markersize),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=10, alpha=0.5),
    ]
    if not plot_std:
        second_legend_handles = second_legend_handles[:2]
        second_legend_labels = second_legend_labels[:2]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper center' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower center'
    ax_twin.legend(second_legend_handles[1:], second_legend_labels[1:], loc=loc)
    ax.yaxis.grid()

    # Third legend
    models = model_to_scenarios.keys()
    third_legend_labels = models
    third_legend_handles = [plt.Line2D([0], [0], marker=get_marker(model), linestyle='',
                                       color='k', markerfacecolor='w', markersize=markersize) for model in models]

    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper right' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower right'
    ax_twin.legend(third_legend_handles, third_legend_labels, loc=loc)

    show_or_save_plot(f'comparison_between_RCP_and_SSP_scenarios_{variable_name}', show=show)




if __name__ == '__main__':
    for name in ['NPP', 'SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']:
        main_plot_rcp_and_ssp(name, plot_std=False, show=True)