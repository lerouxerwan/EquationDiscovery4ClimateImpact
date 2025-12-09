from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from plot.dataset.plot_dataset import plot_values_feature, plot_values_target, plot_values
from utils.utils_plot import get_subplots, show_or_save_plot


def main_plot_data(show=False):
    # Load axis and dataset
    fig, (ax1, ax2) = get_subplots(1, 2, wspace=0.15)
    dataset =  get_dataset()

    # Add two plots
    column_index = dataset.X_variable_names.index('SST_JJA')
    sst_train = dataset.X_train[:, column_index] - 273.15
    sst_test = dataset.X_test[:, column_index] - 273.15
    label_sst = dataset.X_labels[column_index].replace('(K)', '($^o$C)')
    plot_values(ax1, dataset, sst_train, sst_test, label_sst)
    plot_values_target(ax2, dataset)

    # Some additional things for Slides
    # for ax in [ax1, ax2]:
    #     size = 13
    #     ax.xaxis.label.set_size(size)
    #     ax.yaxis.label.set_size(size)
    show_or_save_plot('data', show)



def main_plot_some_features(show=False):
    # Load axis and dataset
    dataset =  get_dataset()
    print(dataset.X_variable_names)
    variable_names = ['SSS_AnnSea', 'SSS_JJA', 'SSS_SON', 'SST_DJF', 'SST_MAM']
    for variable_name in variable_names:
        ax = plt.gca()
        plot_values_feature(ax, dataset, dataset.X_variable_names.index(variable_name),
                            False, False, False)
        show_or_save_plot(f'data_{variable_name}', show)


if __name__ == '__main__':
    main_plot_data(show=True)
    # main_plot_some_features(show=False)