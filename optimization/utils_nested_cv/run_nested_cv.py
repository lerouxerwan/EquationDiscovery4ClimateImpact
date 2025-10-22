import os
import os.path as op

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import KFold

from data.utils_dataset.dataset import Dataset
from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_dataset.utils_validation_split import get_validation_mask
from optimization.optimization import Optimization
from optimization.utils_optimization import get_loss
from plot.utils_metric.metric import Metric, metric_to_str
from utils.utils_log import log_info
from utils.utils_path import NESTED_CV_PATH
from utils.utils_run import random_seed


def run_nested_cv(dataset: Dataset, optimization: Optimization) -> ndarray:
    metric = Metric.RMSE
    filename = f'{metric_to_str[metric]}_test.txt'
    filepath = op.join(NESTED_CV_PATH, dataset.hash, optimization.opt_id, filename)
    if op.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)
        rmse_test_list = df['RMSE'].values
    else:
        rmse_test_list = _run_nested_cv(dataset, optimization, metric)
        df = pd.DataFrame.from_dict({'RMSE': rmse_test_list})
        dirname = op.dirname(filepath)
        if not op.exists(dirname):
            os.makedirs(dirname)
        df.to_csv(filepath)
    return rmse_test_list


def _run_nested_cv(dataset: Dataset, optimization: Optimization, metric: Metric) :
    """Nested cv on the train set. Several folds are created randomly from the train set.
    Each fold is considered successively as the test set while other folds are used as train & validation sets.
    For each fold, the validation set is built using the validation_size and validation_split as the original dataset
    Finally the algorithm returns the list of RMSE test, the length of the list equals the number of folds."""
    # Consider only datapoints from the train set (and exclude datapoints from the validation set and test set)
    X, y = get_X_and_y(dataset.X_train, dataset.y_train, dataset.validation_mask, validation_set=False)
    # Run outer loop
    rmse_test_list = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    for j ,(train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        validation_mask = get_validation_mask(y_train, dataset.validation_size, dataset.validation_split)
        rmse_test = get_loss(optimization, X_train, y_train, validation_mask,
                             dataset.X_variable_names, dataset.X_units, dataset.y_units,
                             metric, X_test, y_test)
        log_info(f'RMSE test for the fold #{j}: {rmse_test}')
        rmse_test_list.append(rmse_test)
    rmse_test_list = np.array(rmse_test_list)
    log_info(f'Summary of nested cv for {optimization.name}:')
    log_info(f' {np.mean(rmse_test_list)} ({np.min(rmse_test_list)}, {np.max(rmse_test_list)})')
    return rmse_test_list

