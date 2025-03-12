from itertools import chain
from operator import itemgetter
from typing import Any

import joblib
import numpy as np


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
    return str(joblib.hash(tuple(chain.from_iterable(tuples))))

def get_key_for_cache_duplicate(X: np.ndarray, y: np.ndarray, threshold: float) -> str:
    return get_hash_str(X, y, [threshold])


def get_key_for_cache_fit(X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> str:
    return get_hash_str(X, y, get_hash_params(params))

def get_hash_params(params: dict[str, Any]) -> list[tuple[Any] | Any]:
    """Summarize all parameters as list (but do not include model_selection_threshold and logger_spec)"""
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





