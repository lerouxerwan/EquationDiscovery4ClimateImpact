from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_bayesian.optimization_bayesian_zoo import OptimizationBayesian_500
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom
from optimization.utils_params.utils_param_name_to_values import ParamNameToValues

validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE, ValidationSplit.START, ValidationSplit.END][:3]
validation_sizes = [0.2, 0.25, 0.3]

# opt_type = OptimizationBayesian_500
# opt_label = 'Bayesian optimization (500 samples)'

opt_type = OptimizationPipelineRandom
opt_label = 'Random optimization (500 samples)'

def get_opt(opt_type: type = opt_type):
    assert issubclass(opt_type, Optimization)
    n_jobs = -1 if opt_type == OptimizationPipelineRandom else 1
    return opt_type('best', ParamNameToValues.DEFAULT_CENTRED_WO_OPERATORS,
                    n_jobs=n_jobs, timeout_in_seconds=60 * 60, interpretable_mode=True)


