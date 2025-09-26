import numpy as np
from sklearn.model_selection import KFold

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_dataset.utils_validation_split import get_validation_mask
from data.utils_dataset.validation_split import ValidationSplit
from optimization.optimization import Optimization
from optimization.optimization_baseline import optimization_baseline_with_validated_model_selection, \
    optimization_baseline_with_best_model_selection
from plot.utils_metric.metric import Metric
from utils.utils_log import log_info
from utils.utils_run import random_seed


def run_nested_cv(dataset: Dataset, optimization: Optimization, fast: bool = False):
    # Consider only datapoints from the train set (and exclude datapoints from the validation set and test set)
    X, y = get_X_and_y(dataset.X_train, dataset.y_train, dataset.validation_mask, validation_set=False)
    # Run outer loop
    rmse_test_list = []
    n_splits = 2 if fast else 10
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for j ,(train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        validation_mask = get_validation_mask(y_train, dataset.validation_size, dataset.validation_split)
        top_emulator = optimization.get_top_emulator(X_train, y_train, validation_mask,
                                                     dataset.X_variables_names, dataset.X_units, dataset.y_units)
        rmse_test = top_emulator.compute_loss(X_test, y_test, Metric.RMSE)
        log_info(f'RMSE test for the fold #{j}: {rmse_test}')
        rmse_test_list.append(rmse_test)
    rmse_test_list = np.array(rmse_test_list)
    log_info('Summary of nested cv:')
    log_info(f' {np.mean(rmse_test_list)} ({np.min(rmse_test_list)}, {np.max(rmse_test_list)})')

if __name__ == '__main__':
    dataset = Dataset("NPP_season.csv", "RCP85", "RCP45", 0.2, ValidationSplit.QUANTILE_WITH_BINNING)
    opt = optimization_baseline_with_best_model_selection
    opt = optimization_baseline_with_validated_model_selection
    run_nested_cv(dataset, opt, True)