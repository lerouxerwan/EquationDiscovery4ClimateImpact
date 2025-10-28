import numpy as np
from matplotlib import pyplot as plt

from data.utils_dataset.npp_season_v1 import get_dataset
from optimization.optimization import Optimization
from optimization.utils_nested_cv.run_nested_cv import run_nested_cv
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_plot import show_or_save_plot


def get_opt_list(model_selection='best', param_name_to_values=ParamNameToValues.DEFAULT_CENTRED) -> list[Optimization]:
    opt_list: list[Optimization] = []
    return opt_list

def main_plot_mean_rmse_consistency(fast: bool):
    dataset = get_dataset()
    opt_list = get_opt_list('best')
    if fast:
        opt_list = opt_list[:]
    true_rmse_test_list = []
    mean_rmse_test_list = []
    for opt in opt_list:
        true_rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                  dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                  dataset.X_test, dataset.y_test)
        true_rmse_test_list.append(true_rmse_test)
        mean_rmse_test_list.append(np.mean(run_nested_cv(dataset, opt)))
    ax = plt.gca()
    ax.scatter(true_rmse_test_list, mean_rmse_test_list, marker='x')
    ax.set_xlabel('RMSE test')
    ax.set_ylabel('Mean RMSE test from nested CV')
    show_or_save_plot('rmse_consistency', show=fast)



if __name__ == '__main__':
    main_plot_mean_rmse_consistency(fast=True)