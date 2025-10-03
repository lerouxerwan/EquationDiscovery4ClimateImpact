from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_baseline import optimization_baseline_with_validated_model_selection
from optimization.optimization_marginal_search import OptimizationMarginalSearch
from optimization.utils_nested_cv import run_nested_cv
from optimization.utils_params.utils_params_values import ParamsValues

if __name__ == '__main__':
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    # opt = optimization_baseline_with_best_model_selection
    # opt = optimization_baseline_with_validated_model_selection
    # opt = OptimizationMarginalSearch('best', 'warmup_maxsize_by')
    for model_selection in ['best', 'validated']:
        opt = OptimizationMarginalSearch(model_selection, ParamsValues.DEFAULT_CENTRED, 'niterations')
        run_nested_cv(dataset, opt)