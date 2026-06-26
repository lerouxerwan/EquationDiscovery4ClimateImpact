import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from plot.by_rcp.utils_rcp import rcp_name_to_color, get_rcp_label
from plot.by_rcp.utils_ssp import ssp_name_to_color, get_ssp_label
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.ssp_scenarios.get_df_rcsm4 import get_df_rcsm4
from projects.ssp_scenarios.get_df_ssp import get_df_rcsm6b
from utils.utils_plot import show_or_save_plot

def _get_label(model: str, scenario: str) -> str:
    if scenario == "HIST":
        scenario_label = "HIST"
    else:
        scenario_label = get_rcp_label(scenario) if scenario.startswith('RCP') else get_ssp_label(scenario)
    return f'{model}_{scenario_label}'

def _get_color(model: str, scenario: str) -> str:
    model_and_scenario_to_color = {
        ('RCSM4', 'HIST'): 'k',
        ('RCSM4', 'RCP45'): 'tab:purple',
        ('RCSM4', 'RCP85'): 'r',

        ('RCSM6B', 'SSP370'): 'gold',
        ('RCSM6B', 'SSP585'): 'darkred',
    }
    return model_and_scenario_to_color[(model, scenario)]



def _get_df(model: str, scenario: str) -> pd.DataFrame:
    if model == 'RCSM4':
        return get_df_rcsm4(scenario)
    elif model == 'RCSM6':
        raise NotImplementedError
    elif model == 'RCSM6B':
        return get_df_rcsm6b(scenario)
    else:
        raise ValueError(f"Model {model} not supported")


def main_plot_rcp_and_ssp(variable_name: str, show: bool = False):
    window_size = 30
    ax = plt.gca()
    model_to_scenarios = {
        'RCSM4': ['HIST', 'RCP45', 'RCP85'],
        'RCSM6B': ['SSP370', 'SSP585'],
    }
    for model, scenarios in model_to_scenarios.items():
        for scenario in scenarios:
            df, color, label = _get_df(model, scenario), _get_color(model, scenario), _get_label(model, scenario)
            values = np.array(df[variable_name].values)
            years = df.index.values
            ax.plot(years, values, label=label, color=color, linestyle='', marker='o', markersize=5)
            plot_average_value(ax, color, values, years, window_size)

    # Axes labels
    ax.set_xlabel('Year')
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    if variable_name == 'NPP':
        y_label = get_label(dataset.target_label)
    else:
        y_label = dataset.X_labels[dataset.X_variable_names.index(variable_name)]
    ax.set_ylabel(y_label)
    # First legend
    ax.legend()
    # Second legend
    legend_labels = ['Annual value', f'{window_size}-years average', 'Standard deviation']
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='k', markerfacecolor='w'),
        plt.Line2D([0], [0], marker='', linestyle='-', color='k'),
        plt.Line2D([0], [0], marker='s', linestyle='', color='k', markerfacecolor='k', markersize=10,
                   alpha=0.5),
    ]
    ax_twin = ax.twinx()
    ax_twin.set_yticks([])
    loc = 'upper center' if variable_name in ['SST_MAM', 'SST_DJF'] else 'lower center'
    ax_twin.legend(legend_handles, legend_labels, loc=loc)
    ax.yaxis.grid()

    show_or_save_plot(f'comparison_between_RCP_and_SSP_scenarios_{variable_name}', show=show)




if __name__ == '__main__':
    for name in ['NPP', 'SSS_MAM', 'SST_MAM', 'SST_DJF', 'Shortwave_DJF']:
        main_plot_rcp_and_ssp(name, show=True)