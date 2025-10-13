import sys

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.utils_metric.metric import Metric
from utils.utils_log import log_info


def main():
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 3)]
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED
    else:
        indices = [0, 0]
        # 'populations': [2, 4],
        # param_name_to_values = {'niterations': [2, 4], 'ncycles_per_iteration': [2, 4]}
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED

    print(f'Run with indices={indices}')
    # Load dataset
    validation_size = [0.2, 0.25, 0.3][indices[0]]
    validation_split = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][indices[1]]
    dataset = Dataset("NPP_season_and_annual_season.csv", "RCP85", "RCP45", validation_size, validation_split)

    # Run optimization
    opt = OptimizationPipelineRandom('best', param_name_to_values, n_jobs=-1)
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units,
             Metric.RMSE, dataset.X_test, dataset.y_test)
    log_info(f'RMSE test for top emulator = {rmse_test}')

    # run_nested_cv(dataset, opt)
if __name__ == '__main__':
    main()