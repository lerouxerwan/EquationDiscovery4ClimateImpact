import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import dataset_values_npp_season_v1
from emulator.utils_plots.plot_by_rcp.plot_time_series_climato import _plot_climatological_time_series
from emulator.utils_plots.plot_by_rcp.utils_plot_by_rcp import load_rcp_name_to_list_of_years_and_y_and_color_and_label
from utils.utils_plot import subplots_custom, show_or_save_plot


def plot_data(ax, y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, target_label, validation_mask):
    rcp_name_to_list_of_years_and_y_and_color_and_label = load_rcp_name_to_list_of_years_and_y_and_color_and_label(y_train, y_test, years_train, years_test, rcp_name_train, rcp_name_test, validation_mask)
    _plot_climatological_time_series(ax, rcp_name_to_list_of_years_and_y_and_color_and_label, y_train, target_label, None)
    y_all = np.concat([y_train, y_test], axis=0)
    ymin, ymax = 0.99 * np.min(y_all), 1.01 * np.max(y_all)
    ax.set_ylim(ymin, ymax)

def main_plot_data(show=False):
    # Load axis and dataset
    fig, (ax1, ax2) = subplots_custom(1, 2, wspace=0.15)
    dataset_values =  dataset_values_npp_season_v1
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = dataset_values.values

    # Add two plots
    column_index = X_variables_names.index('SST_{JJA}')
    sst_train = X_train[:, column_index] - 273.15
    sst_test = X_test[:, column_index] - 273.15
    label_sst = X_labels[column_index].replace('(K)', '($^o$C)')
    plot_data(ax1, sst_train, sst_test, years_train, years_test,
              dataset_values.rcp_name_train, dataset_values.rcp_name_test, label_sst, validation_mask)
    plot_data(ax2, y_train, y_test, years_train, years_test,
              dataset_values.rcp_name_train, dataset_values.rcp_name_test, y_labels[0], validation_mask)


    # Some additional things for Slides
    # for ax in [ax1, ax2]:
    #     size = 13
    #     ax.xaxis.label.set_size(size)
    #     ax.yaxis.label.set_size(size)
    show_or_save_plot('data', show)



def main_plot_all_features(show=False):
    # Load axis and dataset
    dataset_values =  dataset_values_npp_season_v1
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = dataset_values.values
    variable_names = ['SSH_{DJF}', 'SSS_{MAM}', 'Shortwave_{DJF}', 'MerWindStr_{MAM}']
    for variable_name in variable_names:
        ax = plt.gca()
        column_index = X_variables_names.index(variable_name)
        label_name = X_labels[column_index]
        print(years_train)
        print(years_test)
        print(label_name)
        plot_data(ax, X_train[:, column_index], X_test[:, column_index], years_train, years_test,
                  dataset_values.rcp_name_train, dataset_values.rcp_name_test, label_name, validation_mask)
        show_or_save_plot(f'data_{variable_name}', show)



def get_summer_sst(X, variable_names):
    column_index = variable_names.index('SST_JJA')
    return X[:, column_index] -  273.15



if __name__ == '__main__':
    # main_plot_data(show=False)
    main_plot_all_features(show=False)