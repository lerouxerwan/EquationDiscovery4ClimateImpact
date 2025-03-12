from operator import itemgetter

import numpy as np
import pandas as pd

from emulator.utils_metric.utils_metric_function import correlation
from utils.utils_log import log_info

def absolute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(correlation(y_true, y_pred)))

def compute_duplicate_mask(X: np.ndarray, y: np.ndarray, threshold: float) -> list[bool]:
    """Compute a boolean mask of length m, where m is the number of columns/features in the matrix X, which indicates
    the features to keep. A feature is discarded, i.e. equal to False in the duplicate mask, it is a duplicate.
    We define a duplicate as a feature Xj such as there exists another feature Xi where:
        1) absolute_correlation(Xj, Xi) > threshold
        2) absolute_correlation(Xi, y) > absolute_correlation(Xj, y)
    """
    X = X.copy() if isinstance(X, np.ndarray) else X.values.copy()
    # Compute correlation of every feature with the target
    feature_index_to_x: dict[int, np.ndarray] = {feature_index: x for feature_index, x in enumerate(X.transpose())
                                                 if isinstance(x, np.ndarray)}
    assert len(feature_index_to_x) == X.shape[1]
    feature_index_to_target_absolute_correlation = {i: absolute_correlation(x, y) for i, x in feature_index_to_x.items()}
    # Sort feature index depending on their target correlation
    sorted_items = sorted(feature_index_to_target_absolute_correlation.items(), key=itemgetter(1), reverse=True)
    sorted_feature_indexes = [item[0] for item in sorted_items]
    # Exclude feature index with an absolute correlation of 1,
    # because it means that this feature is constant, and in this case all the other features will be discarded
    feature_index_to_discard = {feature_index for feature_index, target_absolute_correlation in sorted_items
                                if target_absolute_correlation == 1.0}
    # Main loop to exclude duplicates
    feature_index_to_keep = set()
    for i in sorted_feature_indexes:
        if i not in feature_index_to_discard:
            feature_index_to_keep.add(i)
            xi = feature_index_to_x[i]
            feature_indexes_to_not_check = feature_index_to_keep.union(feature_index_to_discard)
            for j, xj in feature_index_to_x.items():
                if j not in feature_indexes_to_not_check:
                    if absolute_correlation(xi, xj) > threshold:
                        feature_index_to_discard.add(j)
    assert len(feature_index_to_keep.union(feature_index_to_discard)) == X.shape[1]
    log_info(f'feature_index_to_discard={feature_index_to_discard}')
    return [i in feature_index_to_keep for i in range(X.shape[1])]
