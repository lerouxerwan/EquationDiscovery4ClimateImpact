from emulator.emulator import Emulator
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values
from plot.plot_diagnosis import plot_diagnosis
from projects.goy.utils_goy import get_goy_dataset
from utils.utils_log import log_info


def main_optimization_goy(nb_variables: int):
    dataset = get_goy_dataset(nb_variables, with_validation=True)
    param_name_to_values = get_param_name_to_values(ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS)
    assert isinstance(param_name_to_values, dict)
    param_name_to_values['batching'] = [True]
    param_name_to_values['batch_size'] = [100]
    opt = OptimizationPipelineRandom('best', param_name_to_values, n_jobs=4, timeout_in_seconds=60 * 60)
    emulator = opt.get_top_emulator(dataset.X_train, dataset.y_train, validation_mask=dataset.validation_mask,
                 variable_names=dataset.X_variable_names)
    plot_diagnosis(emulator, dataset, show=True)


if __name__ == '__main__':
    main_optimization_goy(nb_variables=10)
