from typing import OrderedDict

from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimimization_random_zoo import OptimizationRandom_10, OptimizationRandom_400
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info
from utils.utils_plot import show_or_save_plot


def plot_compare_split(opt_type: type, validation_size: float, show: bool):
    ax = plt.gca()
    for model_selection in ['best', 'validated'][:1]:
        opt = opt_type(model_selection, ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
        validation_name_to_rmse_test_for_selected_equation = OrderedDict()
        validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:]
        for validation_split in validation_splits:
            dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
            rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                 dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                 dataset.X_test, dataset.y_test)
            validation_name = f'{validation_split} {validation_size}'
            log_info(f'RMSE test = {rmse_test} for {validation_name} {model_selection}')
            validation_name_to_rmse_test_for_selected_equation[validation_name] = rmse_test
        ax.plot(validation_name_to_rmse_test_for_selected_equation.keys(), validation_name_to_rmse_test_for_selected_equation.values(), label=model_selection)
    ax.legend()
    show_or_save_plot(f'compare_split_{opt_type.__name__}_{validation_size}', show)


if __name__ == '__main__':
    for opt_type in [OptimizationMarginal]:
        for validation_size in [0.2, 0.25, 0.3][:]:
            plot_compare_split(opt_type, validation_size, show=True)

