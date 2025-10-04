from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import OptimizationBaseline
from optimization.optimization_double_search import OptimizationDoubleSearch
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_nested_cv.compare_nested_cv import compare_nested_cv
from optimization.utils_params.utils_params_values import ParamsValues, param_names
from plot.plot_diagnosis import plot_diagnosis

dataset= Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)

def main_plot_diagnosis_top_emulator():
    opt = OptimizationDoubleSearch('best', ParamsValues.DEFAULT_CENTRED, 4)
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                    dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_diagnosis(emulator, dataset)

def main_compare_nested_cv():
    opt_list: list[Optimization] = []
    for model_selection in ['best']:
        opt_list.append(OptimizationBaseline(model_selection))
        # Add optimization with marginal search
        for param_name in param_names[:1]:
            opt_list.append(OptimizationMarginalSearch(model_selection, ParamsValues.DEFAULT_CENTRED, param_name))
        # Add optimization with double search
        # for nb_top_hyperparameters in range(1, 5):
        #     opt_list.append(OptimizationDoubleSearch(model_selection, ParamsValues.DEFAULT_CENTRED, nb_top_hyperparameters))
    compare_nested_cv(dataset, opt_list)


if __name__ == '__main__':
    # main_compare_nested_cv()
    main_plot_diagnosis_top_emulator()
