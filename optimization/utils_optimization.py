import os
import os.path as op
from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from data.utils_dataset.utils_validation import get_X_and_y
from data.utils_run.utils_key import get_hash_str
from optimization.optimization import Optimization
from plot.utils_metric.metric import Metric, metric_to_str
from utils.utils_path import OPT_PATH


def get_loss(optimization: Optimization,
             X_train: np.ndarray, y_train: np.ndarray, validation_mask: np.ndarray,
             variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
             y_units: Optional[ArrayLike[str]] = None, metric: Metric = Metric.RMSE,
             X_test: np.ndarray = None, y_test: np.ndarray = None) -> float:
    assert validation_mask is not None
    with_test_data = (X_test is not None) and (y_test is not None)
    if with_test_data:
        name = 'test'
        dataset_folder = get_hash_str(X_train, y_train, validation_mask, X_test, y_test)
    else:
        name = 'val'
        dataset_folder = get_hash_str(X_train, y_train, validation_mask)

    # Compute filepath
    opt_path = op.join(OPT_PATH, dataset_folder, optimization.opt_id)
    filepath = op.join(opt_path, f'{metric_to_str[metric]}_{name}.txt')

    # Load the loss or Compute it and save it
    if op.exists(filepath):
        f = open(filepath, 'r')
        loss = float(f.readline())
        f.close()
    else:
        emulator = optimization.get_top_emulator(X_train, y_train, validation_mask, variable_names, X_units, y_units)
        if with_test_data:
            loss = emulator.compute_selected_loss(X_test, y_test)
        else:
            X_validation, y_validation = get_X_and_y(X_train, y_train, validation_mask, validation_set=True)
            loss = emulator.compute_selected_loss(X_validation, y_validation)
        if not op.exists(opt_path):
            os.makedirs(opt_path)
        f = open(filepath, 'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{}'.format(loss))
        f.close()
    return loss

