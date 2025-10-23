from typing import OrderedDict

from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import optimization_types_500
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.utils_metric.metric import Metric
from utils.utils_plot import show_or_save_plot


def plot_compare_split(opt: Optimization, validation_size: float, show: bool):
    name_to_rmse_test_for_selected_equation = OrderedDict()
    validation_splits = [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END,
                         ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:2]
    for validation_split in validation_splits:
        dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
        rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                             dataset.X_variable_names, dataset.X_units, dataset.y_units,
                             Metric.RMSE, dataset.X_test, dataset.y_test)
        name_to_rmse_test_for_selected_equation[str(validation_split)] = rmse_test
    ax = plt.gca()
    ax.plot(name_to_rmse_test_for_selected_equation.keys(), name_to_rmse_test_for_selected_equation.values())
    show_or_save_plot('loss_vs_complexity', show)


if __name__ == '__main__':
    for opt_type in optimization_types_500[1:2]:
        opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1, timeout_in_seconds=60 * 30)
        for validation_size in [0.2, 0.25, 0.3][:]:
            plot_compare_split(opt, validation_size, show=True)

