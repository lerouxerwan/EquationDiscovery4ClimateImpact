from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import dataset_npp_season_v1
from utils.utils_plot import subplots_custom, show_or_save_plot


def main_plot_data(show=False):
    # Load axis and dataset
    fig, (ax1, ax2) = subplots_custom(1, 2, wspace=0.15)
    dataset =  dataset_npp_season_v1
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = dataset.values

    # Add two plots
    column_index = X_variables_names.index('SST_JJA')
    sst_train = X_train[:, column_index] - 273.15
    sst_test = X_test[:, column_index] - 273.15
    label_sst = X_labels[column_index].replace('(K)', '($^o$C)')
    dataset.plot_values(ax1, sst_train, sst_test, label_sst)
    dataset.plot_values(ax2, y_train, y_test, y_labels[0])

    # Some additional things for Slides
    # for ax in [ax1, ax2]:
    #     size = 13
    #     ax.xaxis.label.set_size(size)
    #     ax.yaxis.label.set_size(size)
    show_or_save_plot('data', show)



def main_plot_all_features(show=False):
    # Load axis and dataset
    dataset =  dataset_npp_season_v1
    (X_train, y_train, X_test, y_test, years_train, years_test,
        X_units, y_units, X_labels, y_labels, X_variables_names, y_variable_names,
        validation_mask) = dataset.values
    variable_names = ['SSH_DJF', 'SSS_MAM', 'Shortwave_DJF', 'MerWindStr_MAM']
    for variable_name in variable_names:
        ax = plt.gca()
        column_index = X_variables_names.index(variable_name)
        label_name = X_labels[column_index]
        dataset.plot_values(ax, X_train[:, column_index], X_test[:, column_index], label_name)
        show_or_save_plot(f'data_{variable_name}', show)


if __name__ == '__main__':
    # main_plot_data(show=False)
    main_plot_all_features(show=False)