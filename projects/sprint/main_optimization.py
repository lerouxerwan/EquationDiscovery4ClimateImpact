from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_400
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues
from utils.utils_log import log_info


def main():
    for opt_type in [OptimizationRandom_400]:
        for model_selection in ['best', 'validated']:
            for validation_split in [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.EXTREME][:]:
                for validation_size in [0.2, 0.25, 0.3][:]:
                    log_info(f'Run {validation_size} {validation_split} {opt_type} ')
                    dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)
                    opt = opt_type(model_selection, ParamNameToValues.DEFAULT_CENTRED, n_jobs=1)
                    rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
                                         dataset.X_variable_names, dataset.X_units, dataset.y_units,
                                         dataset.X_test, dataset.y_test)
                    log_info(f'RMSE test for top emulator = {rmse_test}')

if __name__ == '__main__':
    main()