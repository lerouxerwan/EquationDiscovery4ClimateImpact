from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.optimization_double_search import OptimizationDoubleSearch
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_nested_cv.compare_nested_cv import CompareNestedCV
from optimization.utils_params.utils_params_values import ParamsValues, param_names
from plot.plot_diagnosis import plot_diagnosis

dataset= Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)

def main_plot_diagnosis_top_emulator():
    opt = OptimizationDoubleSearch('best', ParamsValues.DEFAULT_CENTRED, 10)
    emulator = opt.get_top_param_names(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                    dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_diagnosis(emulator, dataset)

def main_compare_nested_cv():
    opt_list: list[Optimization] = [OptimizationBaseline()]
    # Add optimization with marginal search
    # for param_name in param_names:
    #     opt_list.append(OptimizationMarginalSearch(params_values=ParamsValues.DEFAULT_CENTRED,
    #                                                param_name=param_name))
    # Add optimization with double search
    for nb_top_hyperparameters in range(1, 2):
        opt_list.append(OptimizationDoubleSearch(params_values=ParamsValues.DEFAULT_CENTRED,
                                       nb_top_hyperparameters=nb_top_hyperparameters))
    CompareNestedCV(dataset, opt_list).plot()


if __name__ == '__main__':
    main_compare_nested_cv()
