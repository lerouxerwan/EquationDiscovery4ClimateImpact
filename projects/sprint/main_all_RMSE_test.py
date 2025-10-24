from itertools import chain
from typing import OrderedDict

from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_10, OptimizationRandom_400, \
    OptimizationRandom_4
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_all_RMSE_test(opt_type: type, validation_size: float, validation_splits: list[ValidationSplit], show: bool):
    title = f'Compare split for validation size {validation_size} and optimization {opt_type.__name__}'
    ax = plt.gca()
    opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
    validation_name_to_all_rmse_test = OrderedDict()
    for validation_split in validation_splits:
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


if __name__ == '__main__':
    fast = False
    validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:]
    if fast:
        validation_splits = validation_splits[:2]
    # for opt_type in [OptimizationMarginal]:
    for opt_type in [OptimizationRandom_4 if fast else OptimizationRandom_400]:
        for validation_size in [0.2, 0.25, 0.3][:]:
            plot_all_RMSE_test(opt_type, validation_size, validation_splits, show=fast)

