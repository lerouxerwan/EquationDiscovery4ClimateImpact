from itertools import chain
from operator import itemgetter
from typing import Any

import numpy as np
import pandas as pd


def get_hash_str(*iterables) -> str:
    tuples = []
    for iterable in iterables:
        if isinstance(iterable, np.ndarray):
            iterable = iterable.flatten()
        elif isinstance(iterable, list):
            pass
        else:
            raise ValueError(f'iterable {iterable} has type {type(iterable)}')
        tuples.append(tuple(list(iterable)))
    return str(hash(tuple(chain.from_iterable(tuples))))

def get_key_for_cache_duplicate_features(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame,
                                         duplicate_feature_threshold: float) -> str:
    return get_hash_str(X, y, [duplicate_feature_threshold])


def get_key_for_cache_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, params: dict[str, Any]) -> str:
    return get_hash_str(X, y, get_hash_params(params))

def get_hash_params(params: dict[str, Any]) -> list[tuple[Any] | Any]:
    """All parameters except model_selection_threshold"""
    params_to_remove = {'threshold_for_model_selection', 'logger_spec'}
    l = []
    for k,v in sorted(list(params.items()), key=itemgetter(0)):
        if k not in params_to_remove:
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = tuple(get_hash_params(v))
            l.append(v)
    return l





