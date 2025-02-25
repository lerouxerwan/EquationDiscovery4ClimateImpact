from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def get_X_sum_and_y_sum(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> tuple[float, float]:
    """Compute the sum of X and the sum y, to have two numbers that almost surely uniquely characterize a dataset"""
    X_sum, y_sum = (X.sum(), y.sum()) if isinstance(X, np.ndarray) else (X.values.sum(), y.values.sum())
    return float(X_sum), float(y_sum)

def get_non_default_params(estimator: BaseEstimator) -> dict[str, Any]:
    """Return a dictionary that maps each the name of each non default parameter to its non default value"""
    default_params = type(estimator)().get_params()
    return {k: v for k, v in estimator.get_params().items() if v != default_params[k]}