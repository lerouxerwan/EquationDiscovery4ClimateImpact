from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_dataset.validation_split import ValidationSplit
from emulator.emulator import Emulator
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from projects.paper.utils_paper import validation_sizes, validation_splits, opt_type, opt_label
from utils.utils_plot import show_or_save_plot


def get_baseline_rmse_test():
    dataset = get_dataset(validation_size=0., validation_split=ValidationSplit.NONE)
    emulator = Emulator('best', timeout_in_seconds=60 * 60, interpretable_mode=True)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names, X_units=dataset.X_units, y_units=dataset.y_units)
    print(emulator.selected_equation)
    return emulator.compute_loss(dataset.X_test, dataset.y_test)


def main_compare_optimization_methods(show: bool):
    ax = plt.gca()
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]
    # Three bars for each dataset, we plot the first bar for all datasets, then the second barn then third bar
    width, coordinates_list = load_bar_attributes(nb_bars=2, x_values_list=list(range(len(datasets))))
    opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=1, timeout_in_seconds=60 * 60, interpretable_mode=True)
    loss_list = [get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                         dataset.X_variable_names, dataset.X_units, dataset.y_units,
                         dataset.X_test, dataset.y_test) for dataset in datasets]

    # any_dataset = datasets[0]
    # X_validation, y_validation = get_X_and_y(any_dataset.X_train, any_dataset.y_train, any_dataset.validation_mask, validation_set=True)
    # print(y_validation)


    barplot = ax.bar(coordinates_list[0], loss_list, width=width, label=opt_label, facecolor="white", edgecolor='black')
    loss_list_labels = [str(round(loss, 2)) for loss in loss_list]
    ax.bar_label(barplot, labels=loss_list_labels, label_type='edge', padding=1, rotation=90)

    # loss_list = [get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
    #                      dataset.X_variable_names, dataset.X_units, dataset.y_units) for dataset in datasets]
    # barplot = ax.bar(coordinates_list[1], loss_list, width=width, label=opt_label, color='orange')




    # Add line for the baseline with the default hyperparameter
    x_min, x_max = ax.get_xlim()
    y = get_baseline_rmse_test()
    ax.hlines(y, x_min, x_max, label='Baseline with default hyperparameters and no validation set', color='k', linestyle='dashed')

    # Hide labels on xaxis
    ax.axes.xaxis.set_ticklabels([])
    # Set name of the dataset on the middle bar
    ax.set_xticks(np.mean(np.array(coordinates_list), axis=0))
    # xticklabels = [dataset.validation_label.replace(' the historical', '\nthe historical') for dataset in datasets]
    xticklabels = [dataset.validation_label.split(' in the')[0].split(' of the')[0] for dataset in datasets]
    ax.set_xlabel('Validation sets ')
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')

    #  General settings for the plot
    ax.legend(loc='upper right')
    ax.set_ylim(0, 3)
    ax.set_ylabel(f'Root mean square error\non the test set RCP4.5 (gC year$^{-1}$)')
    show_or_save_plot('compare_optimization_method', show)

if __name__ == '__main__':
    main_compare_optimization_methods(False)
    # print(get_baseline_rmse_test())