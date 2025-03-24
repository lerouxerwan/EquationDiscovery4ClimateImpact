import numpy as np

from data.utils_dataset.utils_dataset import load_dataset
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import plot_climatological_time_series, \
    _plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_plot import subplots_custom, show_or_save_plot


def plot_data(ax, y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, target_label, validation_mask, show):
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, validation_mask)
    _plot_climatological_time_series(ax, rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, f"", show)
    y_all = np.concat([y_train, y_test], axis=0)
    ymin, ymax = 0.99 * np.min(y_all), 1.01 * np.max(y_all)
    ax.set_ylim(ymin, ymax)

def main_plot_data(show=False):
    # Load axis and dataset
    fig, (ax1, ax2) = subplots_custom(1, 2, wspace=0.15)
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(filename_dataset_paper)
    # Add two plots
    label1 = 'Sea surface temperature in summer ($^o$C)'
    plot_data(ax1, get_summer_sst(X_train, variable_names), get_summer_sst(X_test, variable_names), years_train, years_test, rcp_name_train, rcp_name_test, label1, validation_mask, show)
    label2 = 'Annual net primary production (gC $\;$ year$^{-1}$)'
    plot_data(ax2, y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, label2, validation_mask, show)
    show_or_save_plot('data', show)

def get_summer_sst(X, variable_names):
    column_index = variable_names.index('Mean_SST_JJA')
    return X[:, column_index] -  273.15


if __name__ == '__main__':
    main_plot_data(show=False)