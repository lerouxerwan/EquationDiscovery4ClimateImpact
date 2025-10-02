import os
import os.path as op
from typing import Optional

import numpy as np
from pysr.utils import ArrayLike

from data.utils_run.utils_key import get_hash_str
from optimization.optimization import Optimization
from plot.utils_metric.metric import Metric, metric_to_str
from utils.utils_path import OPT_PATH


def get_loss_test(optimization: Optimization,
                  X_train: np.ndarray, y_train: np.ndarray, validation_mask: np.ndarray[bool],
                  X_test: np.ndarray, y_test: np.ndarray, metric: Metric,
                  variable_names: Optional[ArrayLike[str]] = None, X_units: Optional[ArrayLike[str]] = None,
                  y_units: Optional[ArrayLike[str]] = None) -> float:
    assert validation_mask is not None
    dataset_folder = get_hash_str(X_train, y_train, validation_mask, X_test, y_test)
    opt_path = op.join(OPT_PATH, dataset_folder, optimization.opt_id)
    filepath = op.join(opt_path, f'{metric_to_str[metric]}.txt')
    if op.exists(opt_path):
        f = open(filepath, 'r')
        loss = float(f.readline())
        f.close()
    else:
        top_emulator = optimization.get_top_emulator(X_train, y_train, validation_mask,
                                                     variable_names, X_units, y_units)
        loss = top_emulator.compute_loss(X_test, y_test, Metric.RMSE)
        if not op.exists(opt_path):
            os.makedirs(opt_path)
        f = open(filepath, 'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{}'.format(loss))
        f.close()
    return loss

