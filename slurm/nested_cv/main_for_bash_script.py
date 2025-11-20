
import sys

from data.utils_dataset.npp_season_v1 import get_dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_bayesian.optimization_bayesian_zoo import OptimizationBayesian_500
from optimization.optimization_marginal.optimization_marginal import OptimizationMarginal
from optimization.optimization_random.optimization_random_zoo import OptimizationRandom_200, OptimizationRandom_5
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom, \
    OptimizationPipelineMarginalRandom, OptimizationPipelineMarginalBayesian
from optimization.utils_optimization import get_loss
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues, get_param_name_to_values
from plot.by_split.plot_loss_vs_complexity import plot_loss_vs_complexity
from plot.plot_diagnosis import plot_diagnosis
from plot.utils_metric.metric import metric_to_str
from utils.utils_log import log_info


def main(index2=0, index3=0):
    if len(sys.argv) > 1:
        indices = [int(sys.argv[i]) for i in range(1, 4)]
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS
        n_jobs = 1
    else:
        indices = [4, index2, index3]
        n_jobs = 1
        param_name_to_values = ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS


    print(f'Run with indices={indices}')
    # Load dataset
    opt_type = [OptimizationMarginal, OptimizationPipelineRandom, OptimizationPipelineMarginalRandom,
                OptimizationPipelineMarginalBayesian, OptimizationBayesian_500][indices[0]]
    validation_size = [0.2, 0.25, 0.3][indices[1]]
    validation_split = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE][indices[2]]
    dataset = get_dataset(validation_size=validation_size, validation_split=validation_split)

    # Run optimization
    opt = opt_type('best', param_name_to_values, n_jobs=n_jobs, timeout_in_seconds=60*60, interpretable_mode=True)
    top_emulator, _  = opt.run(dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units)
    rmse_test = get_loss(opt, dataset.X_train, dataset.y_train, dataset.validation_mask,
             dataset.X_variable_names, dataset.X_units, dataset.y_units,
             dataset.X_test, dataset.y_test)
    log_info(f'{metric_to_str[top_emulator.metric_]} test for top emulator = {rmse_test}')
    # plot_loss_vs_complexity(top_emulator, dataset, show=True)
    # plot_diagnosis(top_emulator, dataset)

if __name__ == '__main__':
    for index2 in [0, 1, 2][:]:
        for index3 in [0, 1, 2][:]:
            main(index2, index3)