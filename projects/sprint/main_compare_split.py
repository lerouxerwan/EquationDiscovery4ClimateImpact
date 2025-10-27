from typing import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_4
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_compare_split(opt_type: type, validation_splits: list[ValidationSplit], show: bool):
    title = f'Compare split for optimization {opt_type.__name__}'
    ax = plt.gca()
    validation_split_to_variable_names_for_selected_equation = OrderedDict()
    validation_split_to_min_RMSE = {validation_split: np.inf for validation_split in validation_splits}
    for validation_size, color in zip([0.2, 0.25, 0.3], ['yellow', 'orange', 'red']):
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
        validation_split_to_rmse_test_for_selected_equation = OrderedDict()
        for validation_split in validation_splits[::-1]:
                print('here', validation_size, validation_split)
                dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
                emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units)
                rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                     dataset.X_test, dataset.y_test)
                log_info(f'RMSE test = {rmse_test} for {validation_split}')
                validation_split_to_rmse_test_for_selected_equation[validation_split] = rmse_test
                if rmse_test < validation_split_to_min_RMSE[validation_split]:
                    validation_split_to_min_RMSE[validation_split] = rmse_test
                    # colored_equation = '\\textcolor{' + color + '}{' + str(emulator.selected_variable_names) + '}'
                    # colored_equation = '\\textcolor{' + color + '}{ok}'
                    colored_equation = '\n'.join(emulator.selected_variable_names)
                    validation_split_to_variable_names_for_selected_equation[validation_split] = colored_equation
        # sorted_validation_splits, sorted_rmse_test_list = sorted(validation_split_to_rmse_test_for_selected_equation.items(), key=lambda c:c[1])
        # ax.plot(validation_splits, sorted_rmse_test_list, label=f'validation_size={int(100 * validation_size)}%')
        ax.plot(validation_split_to_rmse_test_for_selected_equation.keys(), validation_split_to_rmse_test_for_selected_equation.values(),
                label=f'validation_size={int(100 * validation_size)}%')


    ax.set_xlabel('Validation split')
    ax.set_ylabel(f'RMSE test ({dataset.y_units[0]})')
    ax_twin = ax.twiny()
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticklabels(validation_split_to_variable_names_for_selected_equation.values())
    # ax_twin.tick_params(axis='x', labelrotation=90)
    ax.set_ylim((1.4, 2. ))
    ax.grid()
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax_twin.tick_params(axis='x', which='major', labelsize=5)
    show_or_save_plot('_'.join(title.split()), show)


if __name__ == '__main__':
    fast = False
    validation_splits = [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END,
                         ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:]
    for opt_type in [OptimizationRandom_4 if fast else OptimizationRandom_200]:
            plot_compare_split(opt_type, validation_splits, show=False)

