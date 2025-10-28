
import sys

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_4, \
    OptimizationRandom_5
from optimization.optmization_pipeline.optimization_pipeline_zoo_400 import optimization_types_400
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import optimization_types_500
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.plot_diagnosis import plot_diagnosis
from plot.utils_metric.metric import metric_to_str
from utils.utils_log import log_info


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 2)]
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED
        n_jobs = -1
        opt_type = OptimizationRandom_200
    else:
        opt_type = OptimizationRandom_5
        n_jobs = 2
        indices = [0]
        param_name_to_values = {'niterations': [2, 4], 'ncycles_per_iteration': [2, 4]}

    print(f'Run with indices={indices}')
    # Load dataset
    validation_size = [0.2, 0.25, 0.3][indices[0]]
    # validation_split = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][indices[2]]
    # validation_split = [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END][indices[2]]
    dataset = get_dataset(validation_size=validation_size, validation_split=ValidationSplit.QUANTILE_WITH_BINNING)

    # Run optimization
    opt = opt_type('best', param_name_to_values, n_jobs=n_jobs, timeout_in_seconds=60*60, gaussian_fit=True)
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units,
             dataset.X_test, dataset.y_test)
    log_info(f'{metric_to_str[top_emulator.metric_]} test for top emulator = {rmse_test}')

    # run_nested_cv(dataset, opt)
if __name__ == '__main__':
    main()