from itertools import product
from typing import Optional

import numpy as np

from data.utils_dataset.utils_validation_split import get_validation_mask
from projects.paper.utils_paper import validation_splits, validation_sizes


def get_cv(y_train: np.ndarray, years_train: Optional[list[int]] = None, rcp_name_train: Optional[str] = None, fast: bool = False):
    cv = []
    for validation_split, validation_size in product(validation_splits, validation_sizes):
        validation_mask = get_validation_mask(y_train, validation_size, validation_split, years_train, rcp_name_train, ['HIST', 'RCP85'])
        indices = np.arange(len(validation_mask))
        cv.append((indices, indices[validation_mask]))
    if fast:
        cv = cv[:1]
    return cv
