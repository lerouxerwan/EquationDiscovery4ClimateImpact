from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.scenarios.main_plot_scenarios import get_average_reference_value
from projects.scenarios.utils_scenarios import get_color, get_marker, get_years_and_values, \
    get_color_universal, get_scenario_label, START_REFERENCE_YEAR, END_REFERENCE_YEAR
from utils.utils_plot import show_or_save_plot

def get_all_years_and_all_anomalies(model: str, scenario: str, variable_name: str) -> tuple[np.ndarray, np.ndarray]:
    assert scenario != "HIST"
    historical_years, historical_values = get_years_and_values(model, "HIST", variable_name)
    scenario_years, scenario_values = get_years_and_values(model, scenario, variable_name)
    average_reference_value = get_average_reference_value(historical_years, historical_values)
    all_years = np.concatenate([historical_years, scenario_years])
    all_anomalies = np.concatenate([historical_values, scenario_values]) - average_reference_value
    return all_years, all_anomalies

def get_years_and_ratios_for_anomalies(model: str, scenario: str):
    years, npp_anomalies = get_all_years_and_all_anomalies(model, scenario, 'NPP')
    _, npp_anomalies_from_shortwave_term = get_all_years_and_all_anomalies(model, scenario, "NPP_from_shortwave_term")
    ratio_of_anomalies = 100 * (npp_anomalies_from_shortwave_term / npp_anomalies)
    return years, ratio_of_anomalies


def main_ratio_anomalies(show: bool = False):
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

        for j, scenario in enumerate(scenarios):
            # Scenario attributes
            color = get_color(model, scenario)

            #  Extract the years and values for the ratio
            years, ratios_for_anomalies = get_years_and_ratios_for_anomalies(model, scenario)

            # Plot historical ratios
            index_start_scenario = list(years).index(2016)
            if j == 0:
                ax.plot(years[:index_start_scenario], ratios_for_anomalies[:index_start_scenario], label=None,
                        color='k', linestyle='', marker=marker_model, markersize=markersize)
            # Plot scenario ratios
            ax.plot(years[index_start_scenario:], ratios_for_anomalies[index_start_scenario:], label=None, color=color,
                    linestyle='', marker=marker_model, markersize=markersize)

            # Plot average anomaly
            first_year_for_average_value = 1986
            first_index_for_average_value = list(years).index(first_year_for_average_value)
            plot_average_value(ax, color, ratios_for_anomalies[first_index_for_average_value:][::-1],
                               years[first_index_for_average_value:][::-1], window_size, False,
                               linewidth, marker_model, markersize_increase_factor * markersize)

    # Axes labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Anomaly of the Shortwave term of NPP divided by\nthe anomaly for the entire NPP equation (%)')
    lim = 200
    ax.set_ylim(-lim, lim)

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
    loc = 'lower left'
    ax.legend(first_legend_handles, first_legend_labels, loc=loc)

    # Second legend
    second_legend_labels =  ['Annual value', f'{window_size}-years average', 'Standard deviation']
    second_legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w', markersize=markersize),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=markersize_legend, alpha=0.5),
    ]
    second_legend_handles = second_legend_handles[:2]
    second_legend_labels = second_legend_labels[:2]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'lower right'
    ax_twin.legend(second_legend_handles[1:], second_legend_labels[1:], loc=loc)
    ax.yaxis.grid()

    # Third legend
    models = model_to_scenarios.keys()
    third_legend_labels = models
    third_legend_handles = [plt.Line2D([0], [0], marker=get_marker(model), linestyle='',
                                       color='k', markerfacecolor='w', markersize=markersize_legend) for model in models]

    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'lower center'
    ax_twin.legend(third_legend_handles, third_legend_labels, loc=loc)

    plot_name = 'ratio_of_anomalies'
    show_or_save_plot(plot_name, show=show)

if __name__ == '__main__':
    main_ratio_anomalies(show=True)
