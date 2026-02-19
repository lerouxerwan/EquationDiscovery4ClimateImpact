from itertools import product

import numpy as np
from matplotlib import pyplot as plt, patches

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from projects.paper.utils_paper import validation_sizes, validation_splits, opt_type, opt_label
from utils.utils_plot import show_or_save_plot


def get_baseline_losses(dataset: Dataset) -> tuple[float, float]:
    emulator = Emulator('best', timeout_in_seconds=60 * 60, interpretable_mode=True)
    emulator.fit(dataset.X_train, dataset.y_train, dataset.validation_mask,
                 variable_names=dataset.X_variable_names, X_units=dataset.X_units, y_units=dataset.y_units)
    X_validation, y_validation = get_X_and_y(dataset.X_train, dataset.y_train, dataset.validation_mask, validation_set=True)
    return emulator.compute_loss(X_validation, y_validation), emulator.compute_loss(dataset.X_test, dataset.y_test)


def main_compare_optimization_methods_v2(show: bool):
    ax = plt.gca()
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]
    # Three bars for each dataset, we plot the first bar for all datasets, then the second barn then third bar
    width, coordinates_list = load_bar_attributes(nb_bars=2, x_values_list=list(range(len(datasets))))
    opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=1, timeout_in_seconds=60 * 60, interpretable_mode=True)

    validation_loss_list = [get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                         dataset.X_variable_names, dataset.X_units, dataset.y_units) for dataset in datasets]
    test_loss_list = [get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                         dataset.X_variable_names, dataset.X_units, dataset.y_units,
                         dataset.X_test, dataset.y_test) for dataset in datasets]


    colors = ['tab:brown', 'tab:purple']
    loss_list_list = [validation_loss_list, test_loss_list]
    labels = ['on the validation set', 'on the test set']
    for j, (loss_list, color, label) in enumerate(zip(loss_list_list, colors, labels)):
        barplot = ax.bar(coordinates_list[j], loss_list, width=width, label=label, facecolor=color, edgecolor='black')
        loss_list_labels = [str(round(loss, 2)) for loss in loss_list]
        # ax.bar_label(barplot, labels=loss_list_labels, label_type='edge', padding=1, rotation=90)
        ax.bar_label(barplot, labels=loss_list_labels, label_type='center', rotation=90)

    # Add cross for the baseline with the default hyperparameter
    for j, loss_list in enumerate(zip(*[get_baseline_losses(dataset) for dataset in datasets])):
        coordinates = coordinates_list[j]
        ax.plot(coordinates, loss_list, linestyle='', marker='o', color='black')

    #  General settings for the plot
    ax.legend(loc='upper right')
    ax.set_ylim(0, 3)
    ax.set_ylabel(f'Root mean square error (gC year$^{-1}$)')


    # Add second custom legend for the plot
    ax_twin = ax.twinx()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks([])
    legend_handles = [patches.Patch(facecolor='white', edgecolor='k'),       plt.Line2D([0], [0], marker='o', linestyle='', color='k'),]
    # legend_labels = ['Random optimization (500 sets of hyperparameter)', 'Baseline (default set of hyperparameters)']
    legend_labels = ['Validated', 'Baseline']
    ax_twin.legend(legend_handles, legend_labels, loc='upper left', ncol=2)

    # Hide labels on xaxis
    ax.axes.xaxis.set_ticklabels([])
    # Set name of the dataset on the middle bar
    ax.set_xticks(np.mean(np.array(coordinates_list), axis=0))
    # xticklabels = [dataset.validation_label.replace(' the historical', '\nthe historical') for dataset in datasets]
    xticklabels = [dataset.validation_label.split(' in the')[0].split(' of the')[0] for dataset in datasets]
    ax.set_xlabel('Validation sets ')
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')


    show_or_save_plot('compare_optimization_method', show)

if __name__ == '__main__':
    main_compare_optimization_methods_v2(False)