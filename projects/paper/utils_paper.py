from data.utils_dataset.validation_split import ValidationSplit
from optimization.optmization_pipeline.optimization_pipeline_zoo_500 import OptimizationPipelineRandom

validation_splits = [ValidationSplit.RANDOM, ValidationSplit.QUANTILE_WITH_BINNING, ValidationSplit.MIDDLE, ValidationSplit.START, ValidationSplit.END][:3]
validation_sizes = [0.2, 0.25, 0.3]
opt_type = OptimizationPipelineRandom
opt_label = 'Random optimization (500 samples)'
