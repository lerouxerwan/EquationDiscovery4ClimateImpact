import numpy as np

from data.utils_dataset.utils_validation_split import get_validation_mask
from data.utils_dataset.validation_split import ValidationSplit


def compute_validation_mask(validation_split: ValidationSplit):
    y_train = np.array([8, 2, 9, 6, 3, 1, 7, 5, 0, 4])
    validation_size = 0.3
    years_train = list(range(10))
    rcp_name_train = 'RCP85'
    prefixes_train = ['HIST'] * 2 + ['RCP85'] * 8
    return list(get_validation_mask(y_train, validation_size, validation_split, years_train, rcp_name_train, prefixes_train))


def test_validation_split_determinist():
    assert compute_validation_mask(ValidationSplit.START) == [True] * 3 + [False] * 7
    assert compute_validation_mask(ValidationSplit.END) == [False] * 7 + [True] * 3
    assert compute_validation_mask(ValidationSplit.RCP_START) == [False] * 2 + [True] * 3 + [False] * 5
    assert compute_validation_mask(ValidationSplit.SYMMETRICAL) == [False] * 3 + [True] * 3 + [False] * 4
    assert compute_validation_mask(ValidationSplit.MIN) == [False, True] + 3 * [False] + [True] + 2 * [False]+ [True, False]
    assert compute_validation_mask(ValidationSplit.MAX) == [True, False, True] + 3 * [False] + [True] + 3 * [False]
    assert compute_validation_mask(ValidationSplit.EXTREME) == [True, False, True] + 5 * [False] + [True, False]

def test_validation_split_random():
    pass


