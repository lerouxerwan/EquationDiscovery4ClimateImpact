from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineMarginalRandom
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from plot.utils_metric.metric import Metric
from utils.utils_log import log_info

if __name__ == '__main__':
    for opt_type in [OptimizationPipelineMarginalRandom]:
        for validation_size in [0.2, 0.25, 0.3]:
            for validation_split in [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME]:
                dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
                opt = opt_type('best', ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
                top_emulator, _ = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
                                          dataset.X_variable_names, dataset.X_units, dataset.y_units)
                rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                     dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                     Metric.RMSE, dataset.X_test, dataset.y_test)
                log_info(f'RMSE test for top emulator = {rmse_test}')
