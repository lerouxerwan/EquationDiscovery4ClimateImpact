from itertools import chain

import joblib
from numpy import ndarray


def get_hash_str(*iterables) -> str:
    tuples = []
    for iterable in iterables:
        if isinstance(iterable, ndarray):
            iterable = iterable.flatten()
        elif isinstance(iterable, list):
            pass
        else:
            raise ValueError(f'iterable {iterable} has type {type(iterable)}')
        tuples.append(tuple(list(iterable)))
    return str(joblib.hash(tuple(chain.from_iterable(tuples))))






