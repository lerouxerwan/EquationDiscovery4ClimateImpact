import numpy as np
from scipy.stats import norm

from data.utils_dataset.utils_validation_split import get_validation_mask
from utils.utils_run import random_seed
from numpy import ndarray

def load_X_and_y_and_validation_mask_for_test(nb_features=1) -> tuple[ndarray, ndarray, ndarray]:
    X, y = load_X_and_y_for_test(nb_features)
    return X, y, get_validation_mask(y)


def load_X_and_y_for_test(nb_features=1) -> tuple[ndarray, ndarray]:
    n = 100
    X = np.expand_dims(np.arange(n), axis=-1).astype(float)
    y = X[:, 0] ** 2 - 2 * X[:, 0] + 3
    y += norm.rvs(loc=0, scale=1, size=n, random_state=random_seed)
    if nb_features > 1:
        X = np.repeat(X, repeats=nb_features, axis=1)
    return X, y