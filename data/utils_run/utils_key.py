from itertools import chain
from operator import itemgetter
from typing import Any

import joblib
import numpy as np

from emulator.utils_emulator import params_that_do_not_impact_the_fit_results


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
    """Summarize all parameters as list (but do not include params that do not impact the fit results)"""
    params_loop = {k: v for k, v in params.items() if k not in params_that_do_not_impact_the_fit_results}
    entire_hash_params = []
    for k, v in sorted(list(params_loop.items()), key=itemgetter(0)):
        if isinstance(v ,(float, int)):
            hash_params = (k, v)
        elif isinstance(v, (list, str)):
            hash_params = tuple([k]) + tuple(v)
        elif isinstance(v, dict):
            hash_params =  tuple([k])  + tuple(get_hash_params(v))
        else:
            raise ValueError(f'For the key {k}, type(v)={type(v)} with v={v}')
        entire_hash_params.append(hash_params)
    return entire_hash_params





