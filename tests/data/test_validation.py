import numpy as np

from data.utils_dataset.utils_validation_split import get_validation_mask
from data.utils_dataset.validation_split import ValidationSplit


def compute_validation_mask(validation_split: ValidationSplit, validation_size = 0.3):
    y_train = np.array([8, 2, 9, 6, 3, 1, 7, 5, 0, 4])
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

def test_validation_split_quantile_with_binning():
    validation_mask = compute_validation_mask(ValidationSplit.QUANTILE_WITH_BINNING, validation_size=0.5)
    indices_with_true = [i for i, v in enumerate(validation_mask) if v]
    couples_list = [(0, 2), (1, 4), (3, 6), (5, 8), (7, 9)]
    for i, j in couples_list:
        assert (i in indices_with_true) != (j in indices_with_true)
    validation_mask = compute_validation_mask(ValidationSplit.QUANTILE_WITH_BINNING, validation_size=0.2)
    indices_with_true = [i for i, v in enumerate(validation_mask) if v]
    group_list = [(0, 2, 3, 6, 7), (1, 4, 5, 8, 9)]
    for group in group_list:
        assert sum([i in indices_with_true for i in group]) == 1




