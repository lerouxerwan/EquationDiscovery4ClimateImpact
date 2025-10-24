from itertools import chain
from typing import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_400, \
    OptimizationRandom_4, OptimizationRandom_100, OptimizationRandom_200
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.plot_diagnosis import plot_diagnosis
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_all_RMSE_test(opt_type: type, validation_size: float, validation_splits: list[ValidationSplit], show: bool):
    title = f'Compare split for validation size {validation_size} and optimization {opt_type.__name__}'
    ax = plt.gca()
    opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
    validation_name_to_all_rmse_test = OrderedDict()
    for j, validation_split in enumerate(validation_splits):
        dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
        assert isinstance(opt, Optimization)
        emulators = opt.get_all_emulators(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                        dataset.X_variable_names, dataset.X_units, dataset.y_units)
        all_rmse_test = list(chain.from_iterable([emulator.compute_loss_list(dataset.X_test, dataset.y_test) for emulator in emulators]))
        validation_name = f'{validation_split} {validation_size}'
        validation_name_to_all_rmse_test[validation_name] = all_rmse_test
    positions = range(len(validation_splits))
    ax.violinplot(validation_name_to_all_rmse_test.values(), positions, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(validation_name_to_all_rmse_test.keys())
    ax.set_xlabel('Validation split')
    ax.set_ylabel(f'RMSE test ({dataset.y_units[0]})')
    ax.set_title(title)
    # ax.legend()
    show_or_save_plot('_'.join(title.split()), show)

def main_all_RMSE_test():
    fast = False
    validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:]
    # if fast:
    validation_splits = validation_splits[:2]
    # for opt_type in [OptimizationMarginal]:
    for opt_type in [OptimizationRandom_4 if fast else OptimizationRandom_400]:
        for validation_size in [0.2, 0.25, 0.3][-1:]:
            plot_all_RMSE_test(opt_type, validation_size, validation_splits, show=fast)

def main_best_results_number1():
    dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    opt = OptimizationRandom_200('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
    # index=100 pour Random
    # for index in [67]:
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                        dataset.X_variable_names, dataset.X_units, dataset.y_units)
    # print(min(emulator.compute_loss_list(dataset.X_test, dataset.y_test)))
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset)

def main_baseline():
    dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    opt = OptimizationBaseline('best', None, n_jobs=1)
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, None,
                                    dataset.X_variable_names, dataset.X_units, dataset.y_units)
    # print(min(emulator.compute_loss_list(dataset.X_test, dataset.y_test)))
    # plot_loss_vs_complexity(emulator, dataset, show=True)
    plot_diagnosis(emulator, dataset)


if __name__ == '__main__':
    main_best_results_number1()
    # main_baseline()


