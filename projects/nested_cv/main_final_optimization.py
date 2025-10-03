from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_double_search import OptimizationDoubleSearch
from optimization.utils_params.utils_params_values import ParamsValues
from plot.plot_diagnosis import plot_diagnosis

if __name__ == '__main__':
    for model_selection in ['best', 'validated']:
        dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
        opt = OptimizationDoubleSearch(model_selection, ParamsValues.DEFAULT_CENTRED, 10)
        emulator = opt.get_top_param_names(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                        dataset.X_variable_names, dataset.X_units, dataset.y_units)
        # plot_diagnosis(emulator, dataset)