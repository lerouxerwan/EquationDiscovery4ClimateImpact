import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from plot.by_rcp.utils_plot_by_rcp import plot_average_value
from plot.by_rcp.utils_rcp import rcp_name_to_color, get_rcp_label
from plot.by_rcp.utils_ssp import ssp_name_to_color, get_ssp_label
from plot.by_split.utlis_plot_selected_equation import get_label
from projects.ssp_scenarios.get_df_rcp import get_df_rcp
from projects.ssp_scenarios.get_df_ssp import get_df_ssp
from utils.utils_plot import show_or_save_plot

def get_y_label(variable_name: str) -> str:
    dataset = get_dataset(validation_split=ValidationSplit.NONE)
    if variable_name == 'NPP':
        return get_label(dataset.target_label)
    else:
        return dataset.X_labels[dataset.X_variable_names.index(variable_name)]



def get_infos(scenario: str) -> tuple[pd.DataFrame, str, str]:
    if scenario.startswith('RCP'):
        return get_df_rcp(scenario), rcp_name_to_color[scenario], get_rcp_label(scenario)
    elif scenario.startswith('SSP'):
        return get_df_ssp(scenario), ssp_name_to_color[scenario], get_ssp_label(scenario)
    else:
        raise ValueError(f"Scenario {scenario} not supported")


def main_plot_rcp_and_ssp(variable_name: str, show: bool = False):
    window_size = 30
    ax = plt.gca()
    for scenario in ['RCP45', 'RCP85', 'SSP370', 'SSP585']:
        df, color, label = get_infos(scenario)
        values = np.array(df[variable_name].values)
        years = df.index.values
        ax.plot(years, values, label=label, color=color, linestyle='', marker='o', markersize=5)
        plot_average_value(ax, color, values, years, window_size)
    ax.set_xlabel('Year')
    ax.set_ylabel(get_y_label(variable_name))
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
        main_plot_rcp_and_ssp(name, show=False)