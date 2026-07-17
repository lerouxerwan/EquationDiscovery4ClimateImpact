from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.scenarios.utils_anomaly import START_REFERENCE_YEAR, END_REFERENCE_YEAR
from projects.scenarios.utils_scenarios import get_marker, get_years_and_values, \
    get_color, get_scenario_label, get_linewidth
from utils.utils_plot import show_or_save_plot


def get_average_reference_value(model: str, variable_name: str):
    model_of_reference = 'MEOM_MEAN' if model.startswith('MEOM') else model
    historical_years, historical_values = get_years_and_values(model_of_reference, 'HIST', variable_name)
    historical_year_to_value = dict(zip(historical_years, historical_values))
    reference_years = range(START_REFERENCE_YEAR, END_REFERENCE_YEAR)
    reference_values = [historical_year_to_value[year] for year in reference_years]
    return np.mean(reference_values)

def main_plot_rcp_and_ssp(variable_name: str, plot_anomaly: bool = True, plot_std: bool = True, show: bool = False,
                          plot_average: bool = True):
    # Display parameters
    ax = plt.gca()
    window_size = 20
    markersize_increase_factor = 6
    markersize = 2 if plot_average else 2 * 4
    linestyle = '' if plot_average else '-'
    markersize_average = markersize_increase_factor * markersize

    max_year_for_scenario = None
    min_year_for_historical = None
    min_year_for_historical = 1981
    max_year_for_scenario = 2017

    # Combinations to show
    model_to_scenarios = OrderedDict()
    model_to_scenarios['RCSM4'] = ['RCP45', 'RCP85']
    model_to_scenarios['RCSM6'] = ['SSP585']
    model_to_scenarios['RCSM6B'] = ['SSP370', 'SSP585']
    model_to_scenarios['MEOM_MEAN'] = ['ENS04']
    model_to_scenarios['MEOM_MIN'] = ['ENS04']
    model_to_scenarios['MEOM_MAX'] = ['ENS04']

    for model, scenarios in model_to_scenarios.items():

        # Model attributes
        marker_model = get_marker(model)
        linewidth_model = get_linewidth(model)
        average_reference_value = get_average_reference_value(model, variable_name)

        # Plot historical values (for absolute values or for anomalies)
        historical_years, historical_values = get_years_and_values(model, 'HIST', variable_name)
        historical_plot_values = historical_values - average_reference_value if plot_anomaly else historical_values
        if (min_year_for_historical is not None) and (historical_years[0] < min_year_for_historical):
            index_min_year_for_scenario = list(historical_years).index(min_year_for_historical)
            historical_years = historical_years[index_min_year_for_scenario:]
            historical_plot_values = historical_plot_values[index_min_year_for_scenario:]
        color_historical = get_color("ENS04") if model.startswith("MEOM") else 'k'
        ax.plot(historical_years, historical_plot_values, label=None, color=color_historical,
                linestyle=linestyle, marker=marker_model, markersize=markersize)

        for scenario in scenarios:
            # Scenario attributes
            color = get_color(scenario)

            # Plot scenario values (for absolute values or for anomalies)
            scenario_years, scenario_values = get_years_and_values(model, scenario, variable_name)
            scenario_plot_values = scenario_values - average_reference_value if plot_anomaly else scenario_values
            if (max_year_for_scenario is not None) and (max_year_for_scenario < scenario_years[-1]):
                index_max_year_for_scenario = list(scenario_years).index(max_year_for_scenario)
                scenario_years = scenario_years[:index_max_year_for_scenario+1]
                scenario_plot_values = scenario_plot_values[:index_max_year_for_scenario+1]
            ax.plot(scenario_years, scenario_plot_values, label=None, color=color,
                    linestyle=linestyle, marker=marker_model, markersize=markersize)


            # Plot average values (for absolute values or for anomalies)
            if plot_average:
                first_year_for_average_value = 1986
                first_year_for_average_value = historical_years[0]
                assert historical_years[0] <= first_year_for_average_value
                first_index_for_average_value = list(historical_years).index(first_year_for_average_value)
                all_years = np.concatenate([historical_years[first_index_for_average_value:], scenario_years])
                all_plot_values = np.concatenate([historical_plot_values[first_index_for_average_value:], scenario_plot_values])
                # Compute averages
                shift = window_size // 2
                # WARNING, when we retrieve values for the average, we go one step further to the left.
                window_values_list = [all_plot_values[i - shift: i + shift] for i in range(shift, len(all_years) - shift)]
                averaged_values = [np.mean(window_values) for window_values in window_values_list]
                years_average = all_years[shift:-shift]
                # Plot averaged values
                if model.startswith('MEOM'):
                    # Plot in reversed order, so that the marker is at the end
                    ax.plot(years_average[::-1], averaged_values[::-1], color=color,
                            linewidth=linewidth_model, marker=marker_model,
                            markevery=150, markersize=markersize_average)
                else:
                    # Compute the first year when the computation of average involve a scenario value
                    first_year_where_a_scenario_value_is_contained_in_average = 2015 - shift - 1
                    index_first_year = list(years_average).index(
                        first_year_where_a_scenario_value_is_contained_in_average)
                    # Plot these averaged values with two colors
                    ax.plot(years_average[:index_first_year], averaged_values[:index_first_year], color='k',
                            linewidth=linewidth_model, marker=marker_model, markevery=150, markersize=markersize_average)
                    ax.plot(years_average[index_first_year:][::-1], averaged_values[index_first_year:][::-1], color=color, linewidth=linewidth_model, marker=marker_model,
                            markevery=150, markersize=markersize_average)


    # Axes labels
    ax.set_xlabel('Year')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    if variable_name in ['NPP', 'NPP_without_shortwave_term', 'NPP_from_shortwave_term', 'relative_contribution_NPP_for_shortwave_term']:
        y_label = get_label(dataset.target_label)
        if variable_name == 'NPP_without_shortwave_term':
            y_label = y_label.replace('Annual net primary production', 'Annual net primary production wo shortwave term')
        elif variable_name == 'NPP_from_shortwave_term':
            y_label = y_label.replace('Annual net primary production', 'Annual net primary production from shortwave term')
        elif variable_name == 'relative_contribution_NPP_for_shortwave_term':
            y_label = 'Relative contribution to the NPP from the shortwave term (%)'
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
            label_to_color[get_scenario_label(scenario)] = get_color(scenario)
    first_legend_labels =  list(label_to_color.keys())
    first_legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                       color=color, markerfacecolor=color, markersize=markersize_legend) for color in label_to_color.values()]
    loc = 'upper left' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower left'
    ax.legend(first_legend_handles, first_legend_labels, loc=loc)

    # Second legend
    if plot_average:
        second_legend_labels =  ['Annual value', f'{window_size}-years\naverage', 'Standard deviation']
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
    ncol = 2 if len(model_to_scenarios) > 3 else 1
    ax_twin.legend(third_legend_handles, third_legend_labels, loc=loc, ncol=ncol)

    plot_name = f'{"anomalies" if plot_anomaly else "values"}_{variable_name}'
    plot_name += f'_with{'' if plot_average else 'out'}_{window_size}_years_average'
    show_or_save_plot(plot_name, show=show)




if __name__ == '__main__':
    for name in ['NPP', 'SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']:
    # for name in ['NPP', 'NPP_without_shortwave_term', 'NPP_from_shortwave_term']:
    # for name in ['relative_contribution_NPP_for_shortwave_term']:
        for plot_anomaly in [True, False][:]:
            for plot_average in [True, False][:]:
                main_plot_rcp_and_ssp(name, plot_std=False, show=False,
                                      plot_anomaly=plot_anomaly, plot_average=plot_average)