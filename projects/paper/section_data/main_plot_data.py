import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.utils_dataset import load_dataset
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import _plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from projects.paper.utils_paper import filename_dataset_paper
from utils.utils_plot import subplots_custom, show_or_save_plot


def plot_data(ax, y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, target_label, validation_mask):
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
    plot_data(ax1, get_summer_sst(X_train, variable_names), get_summer_sst(X_test, variable_names), years_train, years_test, rcp_name_train, rcp_name_test, label1, validation_mask)
    plot_data(ax2, y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, target_label, validation_mask)
    # Some additional things for Slides
    # for ax in [ax1, ax2]:
    #     size = 13
    #     ax.xaxis.label.set_size(size)
    #     ax.yaxis.label.set_size(size)
    show_or_save_plot('data', show)



def main_plot_all_features(show=False):
    # Load axis and dataset
    (X_train, y_train, X_test, y_test, X_units, y_units, years_train, years_test, rcp_name_train, rcp_name_test,
     variable_names, target_label, validation_mask) = load_dataset(filename_dataset_paper)
    features = ['SSH_DJF', 'SSS_MAM', 'Shortwave_DJF', 'MerWindStr_MAM']
    labels = ['Mean sea surface height in winter (m)', 'Mean sea surface salinity in spring (-)',
              'Mean net downward shortwave flux in winter (W m$^{-2}$)',
              'Mean meridional wind stress in spring (N m$^{-2}$)']
    for variable_name, label_name in zip(features, labels):
        ax = plt.gca()
        column_index = variable_names.index(variable_name)
        plot_data(ax, X_train[:, column_index], X_test[:, column_index], years_train, years_test, rcp_name_train, rcp_name_test, label_name, validation_mask, show)
        show_or_save_plot(f'data_{variable_name}', show)



def get_summer_sst(X, variable_names):
    column_index = variable_names.index('SST_JJA')
    return X[:, column_index] -  273.15

def get_winter_ssh(X, variable_names):
    column_index = variable_names.index('SSH_DJF')
    return X[:, column_index]


if __name__ == '__main__':
    main_plot_data(show=False)
    # main_plot_all_features(show=False)