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
    entire_hash_params = []
    for k, v in sorted(list(params.items()), key=itemgetter(0)):
        if k not in params_to_remove:
            if isinstance(v ,(float, int)):
                hash_params = (k, v)
            elif isinstance(v, (list, str)):
                hash_params = tuple([k]) + tuple(v)
            elif isinstance(v, dict):
                hash_params =  tuple([k])  + tuple(get_hash_params(v))
            else:
                raise ValueError(f'type(v)={type(v)} with v={v}')
            entire_hash_params.append(hash_params)
    return entire_hash_params





