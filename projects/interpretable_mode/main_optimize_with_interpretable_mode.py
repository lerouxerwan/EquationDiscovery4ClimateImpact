from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optmization_pipeline.optimization_pipeline_zoo_400 import OptimizationPipelineMarginalRandom
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues

opt = OptimizationPipelineMarginalRandom('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS, n_jobs=-1,
                                         interpretable_mode=True, timeout_in_seconds=60*60)
dataset = get_dataset(validation_size=0.3, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                dataset.X_variable_names, dataset.X_units, dataset.y_units)
