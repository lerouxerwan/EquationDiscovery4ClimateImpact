import math
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from data.utils_dataset.validation_split import ValidationSplit
from utils.utils_run import random_seed


def get_validation_mask(y_train: np.ndarray, validation_size: float = 0.3,
                        validation_split: ValidationSplit = ValidationSplit.RANDOM,
                        years_train: Optional[list[int]] = None, rcp_name_train: Optional[str] = None,
                        prefixes_train: Optional[list[str]] = None) -> Optional[np.ndarray]:
    """Compute an array of boolean such that validation_mask[i] = True if the index 'i' is in the validation set"""
    # Initialize validation_mask as False
    length_mask = len(y_train)
    validation_mask = np.zeros(length_mask).astype(bool)
    validation_length = math.ceil(length_mask * validation_size)
    # Set some indices of validation_mask to True (depending on the validation_split considered)
    if validation_split is ValidationSplit.NONE:
        return None
    elif validation_split is ValidationSplit.RANDOM:
        indices = list(range(length_mask))
        indices_validation_set = set(train_test_split(np.array(indices), test_size=validation_size, random_state=random_seed)[1])
        return np.array([i in indices_validation_set for i in indices])
    else:
        # Some checks
        assert (years_train is not None) and (rcp_name_train is not None) and (prefixes_train is not None)
        assert sorted(years_train) == years_train # Ensures the assumption that the rows are in the increasing order
        if validation_split is ValidationSplit.RCP_START:
            first_index_rcp = prefixes_train.index(rcp_name_train)
            validation_mask[first_index_rcp: first_index_rcp + validation_length] = True
        elif validation_split is ValidationSplit.START:
            validation_mask[:validation_length] = True
        elif validation_split is ValidationSplit.END:
            validation_mask[-validation_length:] = True
        elif validation_split is ValidationSplit.SYMMETRICAL:
            start_validation = (length_mask - validation_length) // 2
            validation_mask[start_validation:start_validation+validation_length] = True
        elif validation_split in [ValidationSplit.MIN, ValidationSplit.MAX, ValidationSplit.EXTREME]:
            sorted_index = list(sorted(range(length_mask), key=lambda i: y_train[i]))
            if validation_split is ValidationSplit.MIN:
                indexes_for_validation = sorted_index[:validation_length]
            elif validation_split is ValidationSplit.MAX:
                indexes_for_validation = sorted_index[-validation_length:]
            else:
                indexes_for_validation = sorted_index[:validation_length//2] + sorted_index[-math.ceil(validation_length/2):]
            validation_mask[indexes_for_validation] = True
        else:
            raise NotImplemented
    # Final check on the validation mask
    assert sum(validation_mask) == validation_length
    return validation_mask
