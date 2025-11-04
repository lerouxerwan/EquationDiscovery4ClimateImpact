from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.plot_diagnosis import plot_diagnosis

opt = OptimizationMarginal('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=-1,
                                         interpretable_mode=True, timeout_in_seconds=60*60)
dataset = get_dataset(validation_size=0.2, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                dataset.X_variable_names, dataset.X_units, dataset.y_units)
plot_diagnosis(emulator, dataset)
