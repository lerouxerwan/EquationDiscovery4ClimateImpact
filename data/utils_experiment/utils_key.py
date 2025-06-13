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

def get_hash_params(params: dict[str, Any]) -> list[tuple[Any] | Any]:
    """Summarize all parameters as list (but do not include model_selection and logger_spec)"""
    params_to_remove = {'logger_spec', 'model_selection'}
    l = []
    for k,v in sorted(list(params.items()), key=itemgetter(0)):
        if k not in params_to_remove:
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = tuple(get_hash_params(v))
            l.append(v)
    return l





