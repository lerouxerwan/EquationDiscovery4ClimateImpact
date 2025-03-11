import math
from operator import itemgetter

import numpy as np


def get_weights(y: np.ndarray, weighted_loss_ratio: float) -> np.ndarray:
    """Compute weights for a weighted loss.
    The parameter 'weighted_loss_ratio' indicate the largest weight, while 1.0 will be the smallest weight
    Here, the largest weights are given to most extreme values (minima and maxima),
    while smallest weights are given to the least extreme values the middle
    In between, the weights are linearly spread"""
    couples_sorted = sorted([(i, v) for i, v in enumerate(y)], key=itemgetter(1))
    # Compute weights sorted
    num = math.ceil(len(y) / 2)
    weights_sorted = np.concat([np.linspace(weighted_loss_ratio, 1.0, num=num),
                         np.linspace(1., weighted_loss_ratio, num=num)])
    if len(y) % 2 == 1:
        weights_sorted = np.concat([weights_sorted[:num], weights_sorted[num+1:]])
    # Compute weights
    weights = np.zeros(y.shape)
    for weight, (i, _) in zip(weights_sorted, couples_sorted):
        weights[i] = weight
    return weights


