from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.plot_diagnosis import plot_diagnosis


def main_diagnosis_best_equation():
    dataset = get_dataset(validation_size=0.2, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)
    opt = OptimizationPipelineRandom('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS,
                                     n_jobs=-1, timeout_in_seconds=60 * 60, interpretable_mode=True)
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    plot_diagnosis(top_emulator, dataset)

if __name__ == '__main__':
    main_diagnosis_best_equation()