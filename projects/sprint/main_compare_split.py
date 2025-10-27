from typing import OrderedDict

from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_10, OptimizationRandom_400, \
    OptimizationRandom_200, OptimizationRandom_4
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_compare_split(opt_type: type, validation_size: float, validation_splits: list[ValidationSplit], show: bool):
    title = f'Compare split for validation size {validation_size} and optimization {opt_type.__name__}'
    ax = plt.gca()
    opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1, timeout_in_seconds=60*10)
    validation_name_to_rmse_test_for_selected_equation = OrderedDict()
    validation_name_to_variable_names_for_selected_equation = OrderedDict()
    for validation_split in validation_splits:
        dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
        emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                             dataset.X_variable_names, dataset.X_units, dataset.y_units)
        rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                             dataset.X_variable_names, dataset.X_units, dataset.y_units,
                             dataset.X_test, dataset.y_test)
        validation_name = f'{validation_split} {validation_size}'
        log_info(f'RMSE test = {rmse_test} for {validation_name}')
        validation_name_to_rmse_test_for_selected_equation[validation_name] = rmse_test
        validation_name_to_variable_names_for_selected_equation[validation_name] = str(emulator.selected_variable_names)
    ax.plot(validation_name_to_rmse_test_for_selected_equation.keys(), validation_name_to_rmse_test_for_selected_equation.values())
    ax.set_xlabel('Validation split')
    ax.set_ylabel(f'RMSE test ({dataset.y_units[0]})')
    ax_twin = ax.twiny()
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(validation_name_to_variable_names_for_selected_equation.values())
    # ax_twin.tick_params(axis='x', labelrotation=90)
    ax.set_title(title)
    # ax.legend()
    show_or_save_plot('_'.join(title.split()), show)


if __name__ == '__main__':
    fast = False
    validation_splits = [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END,
                         ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][::1]
    for opt_type in [OptimizationRandom_4 if fast else OptimizationRandom_200]:
        for validation_size in [0.2, 0.25, 0.3][::1]:
            plot_compare_split(opt_type, validation_size, validation_splits, show=fast)

