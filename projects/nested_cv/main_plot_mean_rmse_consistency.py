import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.optimization_marginal_grid import OptimizationMarginalGrid
from optimization.optimization_marginal_grid_then_random import OptimizationMarginalGridThenRandom
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import param_names, ParamNameToValues
from plot.utils_metric.metric import Metric
from utils.utils_plot import show_or_save_plot

dataset= Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)

def get_opt_list(model_selection='best') -> list[Optimization]:
    opt_list: list[Optimization] = []
    opt_list.append(OptimizationBaseline(model_selection))
    for param_name in param_names[:]:
        opt_list.append(OptimizationMarginalGrid(model_selection, ParamNameToValues.DEFAULT_CENTRED, param_name))
    for nb_top_hyperparameters in range(1, 11):
        opt_list.append(
            OptimizationMarginalGridThenRandom(model_selection, ParamNameToValues.DEFAULT_CENTRED, nb_top_hyperparameters))
    return opt_list

def main_plot_mean_rmse_consistency(fast: bool):
    opt_list = get_opt_list('best')
    if fast:
        opt_list = opt_list[:]
    true_rmse_test_list = []
    mean_rmse_test_list = []
    for opt in opt_list:
        true_rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                  dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                  Metric.RMSE, dataset.X_test, dataset.y_test)
        true_rmse_test_list.append(true_rmse_test)
        mean_rmse_test_list.append(np.mean(run_nested_cv(dataset, opt)))
    ax = plt.gca()
    ax.scatter(true_rmse_test_list, mean_rmse_test_list, marker='x')
    ax.set_xlabel('RMSE test')
    ax.set_ylabel('Mean RMSE test from nested CV')
    show_or_save_plot('rmse_consistency', show=fast)



if __name__ == '__main__':
    main_plot_mean_rmse_consistency(fast=True)