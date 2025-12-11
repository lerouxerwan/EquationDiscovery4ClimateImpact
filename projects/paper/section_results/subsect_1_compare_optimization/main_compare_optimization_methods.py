from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit, get_validation_label
from emulator.emulator import Emulator
from optimization.optimization_bayesian.optimization_bayesian_zoo import OptimizationBayesian_400
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_500
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom, OptimizationPipelineMarginalBayesian
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from plot.by_split.utils_axis import set_log_y_axis
from plot.utils_metric.metric import Metric
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot
from itertools import product

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import load_bar_attributes
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def get_baseline_rmse_test():
    dataset = get_dataset(validation_size=0., validation_split=ValidationSplit.NONE)
    emulator = Emulator('best', timeout_in_seconds=60 * 60, interpretable_mode=True)
    emulator.fit(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names, X_units=dataset.X_units, y_units=dataset.y_units)
    return emulator.compute_loss(dataset.X_test, dataset.y_test)


def main_compare_optimization_methods(show: bool, fast: bool = False):
    ax = plt.gca()
    validation_sizes = [0.2, 0.25, 0.3]
    validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE]
    if fast:
        validation_sizes, validation_splits = validation_sizes[:2], validation_splits[:1]
    datasets = [get_dataset(validation_size=validation_size, validation_split=validation_split)
        for validation_split, validation_size in product(validation_splits, validation_sizes)]
    opt_types = [OptimizationRandom_500, OptimizationBayesian_400, OptimizationPipelineMarginalRandom,
                 OptimizationPipelineMarginalBayesian][:1]
    labels = ['Random optimization (500 samples)',
              'Bayesian optimization (400 samples)',
              'Marginal optimization (290 samples)\n'
              'followed by a random optimization (210 samples)\n'
              'on the 5 hyperparameters with best marginal',
            'Marginal optimization (290 samples)\n'
            'followed by a bayesian optimization (210 samples)\n'
            'on the 5 hyperparameters with best marginal'][:1]
    # Three bars for each dataset, we plot the first bar for all datasets, then the second barn then third bar
    width, coordinates_list = load_bar_attributes(nb_bars=len(opt_types), x_values_list=list(range(len(datasets))))
    all_loss_list = []
    for bar_id, (opt_type, label) in enumerate(zip(opt_types, labels)):
        log_info(f"plot bar_id={bar_id}")
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=1, timeout_in_seconds=60 * 60, interpretable_mode=True)
        coordinates = coordinates_list[bar_id]
        loss_list = [get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                             dataset.X_variable_names, dataset.X_units, dataset.y_units,
                             dataset.X_test, dataset.y_test) for dataset in datasets]
        barplot = ax.bar(coordinates, loss_list, width=width, label=label, color=opt.color())
        loss_list_labels = [str(round(loss, 2)) for loss in loss_list]
        ax.bar_label(barplot, labels=loss_list_labels, label_type='edge', padding=1, rotation=90)
        all_loss_list.extend(loss_list)

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
    ax.set_xlabel('Extraction procedure to build the validation set ')
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')

    #  Add y-axis with special scaling
    set_log_y_axis(ax, all_loss_list, datasets[0].target_label, Metric.RMSE)
    #  General settings for the plot
    ax.legend(loc='upper right')
    show_or_save_plot('compare_optimization_method', show)

if __name__ == '__main__':
    main_compare_optimization_methods(False, False)
    # print(get_baseline_rmse_test())