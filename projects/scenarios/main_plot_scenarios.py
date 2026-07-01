from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.scenarios.utils_scenarios import get_color, get_marker, get_years_and_values, \
    get_color_universal, get_scenario_label, START_REFERENCE_YEAR, END_REFERENCE_YEAR
from utils.utils_plot import show_or_save_plot

def get_average_reference_value(historical_years: np.ndarray, historical_values: np.ndarray):
    historical_year_to_value = dict(zip(historical_years, historical_values))
    reference_years = range(START_REFERENCE_YEAR, END_REFERENCE_YEAR)
    reference_values = [historical_year_to_value[year] for year in reference_years]
    return np.mean(reference_values)

def main_plot_rcp_and_ssp(variable_name: str, plot_anomaly: bool = True, plot_std: bool = True, show: bool = False):
    # Display parameters
    ax = plt.gca()
    window_size = 30
    markersize = 2
    markersize_increase_factor = 6
    linewidth = 4

    # Combinations to show
    model_to_scenarios = OrderedDict()
    model_to_scenarios['RCSM4'] = ['RCP45', 'RCP85']
    model_to_scenarios['RCSM6'] = ['SSP585']
    model_to_scenarios['RCSM6B'] = ['SSP370', 'SSP585']

    for model, scenarios in model_to_scenarios.items():

        # Model attributes
        marker_model = get_marker(model)

        # Extract historical years and values
        historical_years, historical_values = get_years_and_values(model, 'HIST', variable_name)

        # Extract the average reference value
        average_reference_value = get_average_reference_value(historical_years, historical_values)

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
            first_year_for_average_value = 1986
            assert historical_years[0] <= first_year_for_average_value
            first_index_for_average_value = list(historical_years).index(first_year_for_average_value)
            all_years = np.concatenate([historical_years[first_index_for_average_value:], scenario_years])
            all_plot_values = np.concatenate([historical_plot_values[first_index_for_average_value:], scenario_plot_values])
            plot_average_value(ax, color, all_plot_values[::-1], all_years[::-1], window_size, plot_std, linewidth, marker_model,
                               markersize_increase_factor * markersize)

    # Axes labels
    ax.set_xlabel('Year')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    if variable_name in ['NPP', 'NPP_without_shortwave_term', 'NPP_from_shortwave_term']:
        y_label = get_label(dataset.target_label)
        if variable_name == 'NPP_without_shortwave_term':
            y_label = y_label.replace('Annual net primary production', 'Annual net primary production wo shortwave term')
        elif variable_name == 'NPP_from_shortwave_term':
            y_label = y_label.replace('Annual net primary production', 'Annual net primary production from shortwave term')
    else:
        y_label = dataset.X_labels[dataset.X_variable_names.index(variable_name)]
    if plot_anomaly:
        y_label = y_label[0].lower() + y_label[1:]
        y_label, unit = y_label.split(' (')
        y_label = (f'Anomaly of {y_label}\n'
                   f'with respect to the reference period {START_REFERENCE_YEAR}-{END_REFERENCE_YEAR} ({unit}')

    ax.set_ylabel(y_label)

    # Three legends
    markersize_legend = 10

    # First legend
    label_to_color = OrderedDict()
    label_to_color['Historical'] = 'k'
    for scenarios in model_to_scenarios.values():
        for scenario in scenarios:
            label_to_color[get_scenario_label(scenario)] = get_color_universal(scenario)
    first_legend_labels =  list(label_to_color.keys())
    first_legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                       color=color, markerfacecolor=color, markersize=markersize_legend) for color in label_to_color.values()]
    loc = 'upper left' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower left'
    ax.legend(first_legend_handles, first_legend_labels, loc=loc)

    # Second legend
    second_legend_labels =  ['Annual value', f'{window_size}-years average', 'Standard deviation']
    second_legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w', markersize=markersize),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=markersize_legend, alpha=0.5),
    ]
    if not plot_std:
        second_legend_handles = second_legend_handles[:2]
        second_legend_labels = second_legend_labels[:2]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper right' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower right'
    ax_twin.legend(second_legend_handles[1:], second_legend_labels[1:], loc=loc)
    ax.yaxis.grid()

    # Third legend
    models = model_to_scenarios.keys()
    third_legend_labels = models
    third_legend_handles = [plt.Line2D([0], [0], marker=get_marker(model), linestyle='',
                                       color='k', markerfacecolor='w', markersize=markersize_legend) for model in models]

    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper center' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower center'
    ax_twin.legend(third_legend_handles, third_legend_labels, loc=loc)

    plot_name = f'{"anomalies" if plot_anomaly else "values"}_{variable_name}'
    show_or_save_plot(plot_name, show=show)




if __name__ == '__main__':
    # for name in ['NPP', 'SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']:
    for name in ['NPP', 'NPP_without_shortwave_term', 'NPP_from_shortwave_term']:
        for plot_anomaly in [True, False]:
            main_plot_rcp_and_ssp(name, plot_std=False, show=False, plot_anomaly=plot_anomaly)