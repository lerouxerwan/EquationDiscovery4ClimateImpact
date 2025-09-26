from data.utils_dataset.dataset import Dataset
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization_baseline import optimization_baseline_with_validated_model_selection
from optimization.utils_nested_cv import run_nested_cv

if __name__ == '__main__':
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    # opt = optimization_baseline_with_best_model_selection
    opt = optimization_baseline_with_validated_model_selection
    run_nested_cv(dataset, opt, False)