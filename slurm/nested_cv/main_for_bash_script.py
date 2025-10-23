
import sys

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optmization_pipeline.optimization_pipeline_zoo_400 import optimization_types_400
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import optimization_types_500
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 4)]
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED
    else:
        indices = [0, 0, 0]
        param_name_to_values = {'niterations': [2, 4], 'ncycles_per_iteration': [2, 4]}

    print(f'Run with indices={indices}')
    # Load dataset
    opt_type = optimization_types_400[indices[0]]
    n_jobs = -1 if indices[0] in {0, 1} else 1
    validation_size = [0.2, 0.25, 0.3][indices[1]]
    # validation_split = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][indices[2]]
    validation_split = [ValidationSplit.START, ValidationSplit.MIDDLE, ValidationSplit.END][indices[2]]
    dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)

    # Run optimization
    opt = opt_type('best', param_name_to_values, n_jobs=n_jobs)
    # top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
    #          dataset.X_variable_names, dataset.X_units, dataset.y_units)
    rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units,
             dataset.X_test, dataset.y_test)
    log_info(f'RMSE test for top emulator = {rmse_test}')

    # run_nested_cv(dataset, opt)
if __name__ == '__main__':
    main()