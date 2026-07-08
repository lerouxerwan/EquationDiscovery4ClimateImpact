from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from projects.scenarios.main_plot_scenarios import get_average_reference_value
from projects.scenarios.utils_scenarios import get_marker, get_years_and_values, \
    get_color, get_scenario_label
from utils.utils_plot import show_or_save_plot


def get_all_years_and_all_absolute_anomalies(model: str, scenario: str, variable_name: str) -> tuple[np.ndarray, np.ndarray]:
    assert scenario != "HIST"
    historical_years, historical_values = get_years_and_values(model, "HIST", variable_name)
    scenario_years, scenario_values = get_years_and_values(model, scenario, variable_name)
    average_reference_value = get_average_reference_value(model, variable_name)
    all_years = np.concatenate([historical_years, scenario_years])
    all_anomalies = np.concatenate([historical_values, scenario_values]) - average_reference_value
    return all_years, np.absolute(all_anomalies)



def get_relative_contribution_in_changes_of_npp_from_shortwave_term(model: str, scenario: str, variable_name: str):
    # Compute absolute anomalies
    years, absolute_anomaly_in_shortwave_djf = get_all_years_and_all_absolute_anomalies(model, scenario, "Shortwave_DJF")
    _, absolute_anomaly_in_sss_mam = get_all_years_and_all_absolute_anomalies(model, scenario, "SSS_MAM")
    _, absolute_anomaly_in_sst_mam = get_all_years_and_all_absolute_anomalies(model, scenario, "SST_MAM")
    _, absolute_anomaly_in_sst_djf = get_all_years_and_all_absolute_anomalies(model, scenario, "SST_DJF")
    # Compute contribution of each term
    contribution_term_sst_mam = 1.03 * absolute_anomaly_in_sst_mam
    contribution_term_sst_djf = 0.086 * absolute_anomaly_in_sst_djf
    contribution_term_sss_mam = 6.2 * absolute_anomaly_in_sss_mam
    contribution_term_shortwave_djf = 0.11 * absolute_anomaly_in_shortwave_djf
    # Compute relative contribution
    if variable_name == 'Shortwave_DJF':
        values = contribution_term_shortwave_djf
    elif variable_name == "SSS_MAM":
        values = contribution_term_sss_mam
    elif variable_name == "SST_MAM":
       values = contribution_term_sst_mam
    elif variable_name == 'SST_DJF':
        values = contribution_term_sst_djf
    else:
        raise ValueError(variable_name)
    values *= 100 / (contribution_term_sss_mam + contribution_term_sst_djf + contribution_term_sst_mam + contribution_term_shortwave_djf)
    return years, values



def main_relative_contribution_of_change(variable_name: str, show: bool = False):
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
    model_to_scenarios['RCSM6B'] = ['SSP370', 'SSP585'][:]

    for model, scenarios in model_to_scenarios.items():

        # Model attributes
        marker_model = get_marker(model)

        for j, scenario in enumerate(scenarios):
            # Scenario attributes
            color = get_color(scenario)

            #  Extract the years and values for the ratio
            years, values = get_relative_contribution_in_changes_of_npp_from_shortwave_term(model, scenario, variable_name)

            # Plot historical relative contribution of change
            index_start_scenario = list(years).index(2015)
            if j == 0:
                ax.plot(years[:index_start_scenario], values[:index_start_scenario], label=None,
                        color='k', linestyle='', marker=marker_model, markersize=markersize)
            # Plot scenario ratios
            ax.plot(years[index_start_scenario:], values[index_start_scenario:], label=None, color=color,
                    linestyle='', marker=marker_model, markersize=markersize)

            # Plot average anomaly
            first_year_for_average_value = 1986
            first_year_for_average_value = years[0]
            first_index_for_average_value = list(years).index(first_year_for_average_value)
            plot_average_value(ax, color, values[first_index_for_average_value:][::-1],
                               years[first_index_for_average_value:][::-1], window_size, False,
                               linewidth, marker_model, markersize_increase_factor * markersize)

    # Axes labels
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Relative contribution in changes from {variable_name} term (%)')

    # Three legends
    markersize_legend = 10

    # First legend
    label_to_color = OrderedDict()
    label_to_color['Historical'] = 'k'
    for scenarios in model_to_scenarios.values():
        for scenario in scenarios:
            label_to_color[get_scenario_label(scenario)] = get_color(scenario)
    first_legend_labels =  list(label_to_color.keys())
    first_legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                       color=color, markerfacecolor=color, markersize=markersize_legend) for color in label_to_color.values()]
    loc = 'upper left'
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
    loc = 'upper right'
    ax_twin.legend(second_legend_handles[1:], second_legend_labels[1:], loc=loc)
    ax.yaxis.grid()

    # Third legend
    models = model_to_scenarios.keys()
    third_legend_labels = models
    third_legend_handles = [plt.Line2D([0], [0], marker=get_marker(model), linestyle='',
                                       color='k', markerfacecolor='w', markersize=markersize_legend) for model in models]

    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper center'
    ax_twin.legend(third_legend_handles, third_legend_labels, loc=loc)

    plot_name = f'relative_contribution_in_changes_for_{variable_name}'
    show_or_save_plot(plot_name, show=show)

if __name__ == '__main__':
    for variable_name in ["Shortwave_DJF", "SSS_MAM", "SST_MAM", "SST_DJF"]:
        main_relative_contribution_of_change(variable_name, show=True)
