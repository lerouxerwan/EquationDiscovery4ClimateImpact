from operator import itemgetter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

def get_key_for_cache_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, params: dict[str, Any]) -> tuple:
    """Create a complex tuple that can be used as a key for a dictionary"""
    return _get_key_for_cache(X, y, get_hash_params(params))

def get_key_for_cache_duplicate_features(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, 
                                         duplicate_feature_threshold: float) -> tuple:
    return _get_key_for_cache(X, y, [duplicate_feature_threshold])

def _get_key_for_cache(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, l: list[Any]) -> tuple:
    return tuple(list(get_X_sum_and_y_sum(X, y)) + l)

def get_X_sum_and_y_sum(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> tuple[float, float]:
    """Compute the sum of X and the sum y, to have two numbers that almost surely uniquely characterize a dataset"""
    X_sum, y_sum = (X.sum(), y.sum()) if isinstance(X, np.ndarray) else (X.values.sum(), y.values.sum())
    return float(X_sum), float(y_sum)

def get_hash_params(params: dict[str, Any]) -> list[tuple[Any] | Any]:
    """All parameters except model_selection_threshold"""
    l = sorted(list(params.items()), key=itemgetter(0))
    l2 = [tuple(v) if isinstance(v, list) else v for k, v in l if k != 'threshold_for_model_selection']
    assert len(l2) == len(l) - 1, 'threshold_for_model_selection attribute must be removed from the list'
    return l2





